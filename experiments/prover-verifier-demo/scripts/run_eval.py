#!/usr/bin/env python3
"""Sweep harness — generate the data the plots and viewer consume (Task 9.1).

Drives one prover ↔ verifier pair per (workload, knob_value) cell. Each
cell:
  1. spawns prover + verifier on free ports (port-file handshake)
  2. POST /workload/start with the workload and its knob
  3. sleeps `--duration` seconds
  4. POST /workload/stop, captures observed_flops_total
  5. POST /traffic/finalize on the verifier
  6. invokes the verdict CLI with --transcript / --traffic-digest /
     --workload-summary
  7. records one row to results.jsonl: workload, knob_value, verdict,
     signals, observed_flops, traffic_size, runtime_s

Determinism: each cell uses a fixed seed (default 0). Re-running with the
same Python build, same knob set, same hardware should produce
byte-identical results.

Usage:
    python3 experiments/prover-verifier-demo/scripts/run_eval.py \\
        --out-dir experiments/prover-verifier-demo/data/eval [--smoke]

`--smoke` runs one knob value per workload (the cheating-trip value for
the adversarial workloads, 0 for benign) so the smoke test stays under
2 minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.proverdet._helpers import (  # noqa: E402  (after sys.path setup)
    http_post_json,
    read_bound_port,
    sandbox_env,
)


# Workload registry: name -> (knob_name, [knob_values for full sweep],
# extra_params, smoke_value).
_KNOB_PARAM = {
    "benign": "seed",
    "mixed_lora": "gradient_steps",
    "lora_loading": "lora_bytes",
}
_SWEEPS: dict[str, dict[str, Any]] = {
    "benign": {
        "knob_values": [0, 1, 2, 3, 4],
        "extra": {"prompts": ["e-1", "e-2"], "use_vllm": False},
        "smoke_value": 0,
    },
    "mixed_lora": {
        "knob_values": [0, 1, 2, 4, 8, 16],
        "extra": {
            "prompts": ["e-1", "e-2"],
            "use_vllm": False,
            "seed": 11,
            "matmul_dim": 8,
        },
        "smoke_value": 8,
    },
    "lora_loading": {
        "knob_values": [0, 4096, 65_536, 262_144, 1_048_576],
        "extra": {
            "prompts": ["e-1"],
            "use_vllm": False,
            "seed": 13,
        },
        "smoke_value": 524_288,
    },
}


class _StaticBytesHandler(BaseHTTPRequestHandler):
    """Serves N bytes of `b'L'` for /lora — backs lora_loading's URL knob."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
        # Honors a single ?n=<int> query if present, else uses Content-Length 0.
        n = 0
        if "?" in self.path:
            _, _, query = self.path.partition("?")
            for kv in query.split("&"):
                k, _, v = kv.partition("=")
                if k == "n" and v.isdigit():
                    n = int(v)
                    break
        body = b"L" * n
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)


class _ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _start_lora_url_server() -> tuple[_ThreadedServer, threading.Thread, str]:
    """Spawn a localhost fake-LoRA bytes server; return (server, thread, base_url)."""
    server = _ThreadedServer(("127.0.0.1", 0), _StaticBytesHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True, name="fake-lora-url")
    t.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    return server, t, base_url


def _spawn_pair(
    work_dir: Path,
) -> tuple[subprocess.Popen[bytes], subprocess.Popen[bytes], int, int]:
    """Spawn verifier + prover with port-file handshake, return both procs + ports."""
    verifier_dir = work_dir / "verifier"
    prover_dir = work_dir / "prover"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    prover_dir.mkdir(parents=True, exist_ok=True)
    verifier_port_file = verifier_dir / "port"
    prover_port_file = prover_dir / "port"

    verifier_log = open(verifier_dir / "server.log", "wb")  # noqa: SIM115
    verifier = subprocess.Popen(
        [
            sys.executable,
            "cmd/verifier_server/main.py",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--port-file",
            str(verifier_port_file),
            "--out-dir",
            str(verifier_dir),
            "--prover-base-url",
            "http://127.0.0.1:1",  # patched to real prover after prover binds
            "--no-scheduler",
        ],
        stdout=verifier_log,
        stderr=verifier_log,
        cwd=str(REPO_ROOT),
        env=sandbox_env(),
    )
    verifier_port = read_bound_port(verifier_port_file, timeout_s=15.0)

    prover_log = open(prover_dir / "server.log", "wb")  # noqa: SIM115
    prover = subprocess.Popen(
        [
            sys.executable,
            "cmd/prover/main.py",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--port-file",
            str(prover_port_file),
            "--run-id",
            "eval-cell",
            "--out-dir",
            str(prover_dir),
            "--verifier-url",
            f"http://127.0.0.1:{verifier_port}",
        ],
        stdout=prover_log,
        stderr=prover_log,
        cwd=str(REPO_ROOT),
        env=sandbox_env(),
    )
    prover_port = read_bound_port(prover_port_file, timeout_s=15.0)
    return verifier, prover, verifier_port, prover_port


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _signals_for_verdict(verdict: dict[str, Any]) -> dict[str, str]:
    """Map verdict.reasons[] to a {signal: pass|fail} summary."""
    reasons = verdict.get("reasons") or []
    sig: dict[str, str] = {
        "replay_correctness": "pass",
        "compute_budget": "pass",
        "bandwidth": "pass",
    }
    for r in reasons:
        rl = r.lower()
        if "compute" in rl or "flop" in rl:
            sig["compute_budget"] = "fail"
        elif "bandwidth" in rl or "traffic" in rl:
            sig["bandwidth"] = "fail"
        elif "replay" in rl:
            sig["replay_correctness"] = "fail"
    return sig


def _run_cell(
    workload: str,
    knob_value: int,
    *,
    duration_s: float,
    lora_url: str | None = None,
) -> dict[str, Any]:
    """Run one (workload, knob_value) cell end-to-end; return a results row."""
    extra = _SWEEPS[workload]["extra"]
    knob_param = _KNOB_PARAM[workload]
    params: dict[str, Any] = {**extra, knob_param: knob_value}
    if workload == "lora_loading":
        if lora_url is None:
            raise RuntimeError("lora_loading cell requires lora_url")
        params["lora_url"] = f"{lora_url}/lora?n={knob_value}"

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"eval-{workload}-{knob_value}-") as tmp:
        work_dir = Path(tmp)
        verifier, prover, verifier_port, prover_port = _spawn_pair(work_dir)
        try:
            status, _ = http_post_json(
                f"http://127.0.0.1:{prover_port}/workload/start",
                {"name": workload, "params": params},
                timeout=15.0,
            )
            if status != 200:
                raise RuntimeError(f"workload/start failed: status={status}")

            time.sleep(duration_s)

            status, stop_body = http_post_json(
                f"http://127.0.0.1:{prover_port}/workload/stop", {}, timeout=30.0
            )
            if status != 200:
                raise RuntimeError(f"workload/stop failed: status={status}")
            workload_summary_path = work_dir / "verifier" / "workload_summary.json"
            workload_summary_path.write_text(json.dumps(stop_body), encoding="utf-8")

            status, fbody = http_post_json(
                f"http://127.0.0.1:{verifier_port}/traffic/finalize",
                {},
                timeout=15.0,
            )
            if status != 200:
                raise RuntimeError(f"traffic/finalize failed: status={status}")
            traffic_size = int(fbody.get("size_bytes", 0))

            verdict_path = work_dir / "verifier" / "verdict.json"
            verdict_proc = subprocess.run(
                [
                    sys.executable,
                    "cmd/verifier_cli/main.py",
                    "--transcript",
                    str(work_dir / "verifier" / "transcript.jsonl"),
                    "--traffic-digest",
                    str(work_dir / "verifier" / "traffic.digest"),
                    "--workload-summary",
                    str(workload_summary_path),
                    "--out",
                    str(verdict_path),
                ],
                cwd=str(REPO_ROOT),
                env=sandbox_env(),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if verdict_proc.returncode != 0:
                raise RuntimeError(f"verdict CLI failed: {verdict_proc.stderr}")
            verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        finally:
            _terminate(prover)
            _terminate(verifier)

    return {
        "workload": workload,
        "knob_param": knob_param,
        "knob_value": knob_value,
        "verdict": verdict["verdict"],
        "reasons": verdict.get("reasons", []),
        "signals": _signals_for_verdict(verdict),
        "observed_flops": int(stop_body.get("observed_flops_total", 0)),
        "claimed_flops": int(stop_body.get("claimed_flops_total", 0)),
        "traffic_size": traffic_size,
        "runtime_s": round(time.monotonic() - started, 3),
    }


def _knob_values_for(workload: str, *, smoke: bool) -> list[int]:
    spec = _SWEEPS[workload]
    if smoke:
        return [int(spec["smoke_value"])]
    return [int(v) for v in spec["knob_values"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prover-verifier-demo eval sweep")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "prover-verifier-demo" / "data" / "eval",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One knob value per workload (CI-friendly).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "results.jsonl"
    duration_s = 1.5 if args.smoke else args.duration

    lora_server, _t, lora_url = _start_lora_url_server()
    try:
        with results_path.open("w", encoding="utf-8") as f:
            for workload in ("benign", "mixed_lora", "lora_loading"):
                for knob_value in _knob_values_for(workload, smoke=args.smoke):
                    print(
                        f"[eval] workload={workload} knob={knob_value} duration={duration_s:.1f}s",
                        flush=True,
                    )
                    row = _run_cell(workload, knob_value, duration_s=duration_s, lora_url=lora_url)
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    print(
                        f"[eval]   verdict={row['verdict']} signals={row['signals']} "
                        f"observed_flops={row['observed_flops']} "
                        f"traffic_size={row['traffic_size']}",
                        flush=True,
                    )
    finally:
        lora_server.shutdown()
        lora_server.server_close()

    print(f"[eval] wrote {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
