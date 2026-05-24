#!/usr/bin/env python3
"""Demo driver — spawns prover + verifier, runs three scenarios, asserts verdicts.

This is the headline deliverable (Task 10.1). The shell script `demo.sh`
is thin; everything that matters happens here.

Each scenario:
  1. POST /workload/start to the prover with fixed cheating-trip params
  2. Live-tail the verifier transcript (one line per /graph, /replay,
     /traffic, /replay/verdict event) for `--per-scenario` seconds
  3. POST /workload/stop, capture observed/claimed totals
  4. POST /traffic/finalize on the verifier
  5. Run modules/attestation/verifier_cli with --transcript / --traffic-digest /
     --workload-summary, capture verdict
  6. Compare against expected; record outcome

Exit 0 iff all three actual verdicts match expected.

Usage:
    python3 experiments/prover-verifier-demo/scripts/demo_driver.py \\
        [--per-scenario 5.0] [--remote]

Env (only read in --remote mode):
    PROVER_HOST, PROVER_PORT, VERIFIER_HOST, VERIFIER_PORT
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.proverdet._helpers import (  # noqa: E402  (after sys.path setup)
    http_get_json,
    http_post_json,
    read_bound_port,
    sandbox_env,
)


_SCENARIOS = [
    {
        "name": "benign",
        "title": "Scenario 1: benign inference",
        "params": {"prompts": ["d-1", "d-2"], "use_vllm": False, "seed": 0},
        "expected": "inference",
    },
    {
        "name": "mixed_lora",
        "title": "Scenario 2: mixed inference + LoRA training",
        "params": {
            "prompts": ["d-1", "d-2"],
            "use_vllm": False,
            "seed": 11,
            "matmul_dim": 8,
            "gradient_steps": 8,
        },
        "expected": "training_or_exfil",
    },
    {
        "name": "lora_loading",
        "title": "Scenario 3: LoRA loading",
        "params": {
            "prompts": ["d-1"],
            "use_vllm": False,
            "seed": 13,
            "lora_bytes": 512_000,
            # lora_url is filled in at runtime once the fake-bytes server is up
        },
        "expected": "training_or_exfil",
    },
]


# --------------- fake-LoRA bytes server (for lora_loading) ---------------


class _StaticBytesHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
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


def _start_lora_url_server() -> tuple[_ThreadedServer, str]:
    server = _ThreadedServer(("127.0.0.1", 0), _StaticBytesHandler)
    threading.Thread(target=server.serve_forever, daemon=True, name="demo-fake-lora").start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    return server, base_url


# --------------- transcript tailer (one-line summaries) -----------------


def _format_transcript_line(line: str) -> str | None:
    """Map a transcript JSONL line to a human one-liner, or None to skip."""
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        return None
    direction = e.get("direction", "")
    endpoint = e.get("endpoint", "")
    status = e.get("status_code")
    arrow = "verifier → prover" if direction == "sent" else "prover → verifier"
    if endpoint.startswith("/replay/verdict/"):
        # The /replay/verdict line only appears on the receiving side; skip
        # the (non-existent) sent half. `status==200` ↔ Freivalds + erasure
        # both passed.
        if direction != "received":
            return None
        replay_id = endpoint[len("/replay/verdict/") :]
        outcome = "pass" if status == 200 else "fail"
        return f"verifier verified replay {replay_id}: {outcome}"
    if endpoint == "/graph":
        return f"{arrow}: GET /graph" + (f" → {status}" if status else "")
    if endpoint == "/replay":
        return f"{arrow}: POST /replay" + (f" → {status}" if status else "")
    if endpoint == "/traffic":
        # Skip per-frame chatter — the tailer prints one /traffic line per
        # frame otherwise, drowning out the more interesting events.
        return None
    if endpoint == "/traffic/finalize":
        return f"{arrow}: POST /traffic/finalize → {status}"
    return None


def _tail_transcript(transcript_path: Path, *, started_at: float, deadline: float) -> None:
    """Stream new lines from `transcript_path` until `deadline`. Best-effort."""
    f = None
    pos = 0
    last_open_attempt = 0.0
    while time.monotonic() < deadline:
        if f is None and time.monotonic() - last_open_attempt > 0.2:
            last_open_attempt = time.monotonic()
            if transcript_path.exists():
                f = transcript_path.open("r", encoding="utf-8")
                f.seek(0, 2)  # tail mode
                pos = f.tell()
        if f is not None:
            f.seek(pos)
            chunk = f.read()
            pos = f.tell()
            if chunk:
                for line in chunk.splitlines():
                    formatted = _format_transcript_line(line)
                    if formatted is None:
                        continue
                    t = time.monotonic() - started_at
                    print(f"[t={t:4.1f}s] {formatted}", flush=True)
        time.sleep(0.1)
    if f is not None:
        f.close()


# --------------- server lifecycle (local mode) --------------------------


def _pick_free_port() -> int:
    """Bind to (127.0.0.1, 0), grab the assigned port, close. There's a
    small TOCTOU window between close and the subprocess re-binding the
    same port, but the OS won't rapidly reuse a freshly-released ephemeral
    port — good enough for a single-host demo, and lets us know both
    URLs up front so the verifier scheduler can run immediately."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_servers(
    prover_dir: Path, verifier_dir: Path, *, log_dir: Path
) -> tuple[subprocess.Popen[bytes], subprocess.Popen[bytes], int, int]:
    """Spawn prover + verifier with cross-URLs already known.

    Pre-picks both ports so we can pass each side the other's URL at
    launch time (verifier needs `--prover-base-url` for its scheduler;
    prover needs `--verifier-url` for its traffic publisher). With both
    URLs known, the verifier scheduler runs by default and the
    transcript narration recovers `/graph` + `/replay` events.
    """
    verifier_port = _pick_free_port()
    prover_port = _pick_free_port()
    verifier_port_file = verifier_dir / "port"
    prover_port_file = prover_dir / "port"

    prover_log = open(log_dir / "prover.log", "wb")  # noqa: SIM115
    prover = subprocess.Popen(
        [
            sys.executable,
            "modules/attestation/prover/main.py",
            "--host",
            "127.0.0.1",
            "--port",
            str(prover_port),
            "--port-file",
            str(prover_port_file),
            "--run-id",
            "demo",
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
    # Wait for the prover to bind before starting the verifier — the
    # verifier's scheduler hits /graph immediately and we want it to find
    # the prover up rather than logging spurious connection-refused errors.
    read_bound_port(prover_port_file, timeout_s=15.0)

    verifier_log = open(log_dir / "verifier.log", "wb")  # noqa: SIM115
    verifier = subprocess.Popen(
        [
            sys.executable,
            "modules/attestation/verifier_server/main.py",
            "--host",
            "127.0.0.1",
            "--port",
            str(verifier_port),
            "--port-file",
            str(verifier_port_file),
            "--out-dir",
            str(verifier_dir),
            "--prover-base-url",
            f"http://127.0.0.1:{prover_port}",
            "--graph-period-ms",
            "500",
            "--replay-period-ms",
            "1000",
        ],
        stdout=verifier_log,
        stderr=verifier_log,
        cwd=str(REPO_ROOT),
        env=sandbox_env(),
    )
    read_bound_port(verifier_port_file, timeout_s=15.0)
    return verifier, prover, verifier_port, prover_port


def _await_health(host: str, port: int, *, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        with suppress(Exception):
            status, _ = http_get_json(f"http://{host}:{port}/health", timeout=1.5)
            if status == 200:
                return
        time.sleep(0.2)
    raise TimeoutError(f"{host}:{port}/health never returned 200 ({last_err})")


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


# --------------- one scenario --------------------------------------------


def _signals_summary(verdict_obj: dict[str, Any]) -> str:
    reasons = verdict_obj.get("reasons", []) or []
    sig = {"replay_correctness": "pass", "compute_budget": "pass", "bandwidth": "pass"}
    for r in reasons:
        rl = r.lower()
        if "compute" in rl or "flop" in rl:
            sig["compute_budget"] = "fail"
        elif "bandwidth" in rl or "traffic" in rl:
            sig["bandwidth"] = "fail"
        elif "replay" in rl:
            sig["replay_correctness"] = "fail"
    return ", ".join(f"{k}={v}" for k, v in sig.items())


def _run_scenario(
    scenario: dict[str, Any],
    *,
    prover_url: str,
    verifier_url: str,
    verifier_dir: Path,
    duration_s: float,
    lora_url: str,
) -> tuple[str, dict[str, Any]]:
    print(f"\n--- {scenario['title']} ---", flush=True)
    params = dict(scenario["params"])
    if scenario["name"] == "lora_loading":
        params["lora_url"] = f"{lora_url}/lora?n={params['lora_bytes']}"

    transcript_path = verifier_dir / "transcript.jsonl"
    deadline = time.monotonic() + duration_s

    started = time.monotonic()
    status, _ = http_post_json(
        f"{prover_url}/workload/start",
        {"name": scenario["name"], "params": params},
        timeout=15.0,
    )
    if status != 200:
        raise RuntimeError(f"workload/start status={status}")

    tailer = threading.Thread(
        target=_tail_transcript,
        kwargs={
            "transcript_path": transcript_path,
            "started_at": started,
            "deadline": deadline,
        },
        daemon=True,
        name=f"tail-{scenario['name']}",
    )
    tailer.start()
    while time.monotonic() < deadline:
        time.sleep(0.05)
    tailer.join(timeout=2.0)

    print(f"[t={duration_s:4.1f}s] stopping workload", flush=True)
    status, stop_body = http_post_json(f"{prover_url}/workload/stop", {}, timeout=30.0)
    if status != 200:
        raise RuntimeError(f"workload/stop status={status}")
    workload_summary_path = verifier_dir / f"workload_summary_{scenario['name']}.json"
    workload_summary_path.write_text(json.dumps(stop_body), encoding="utf-8")

    status, _ = http_post_json(f"{verifier_url}/traffic/finalize", {}, timeout=15.0)
    if status != 200:
        raise RuntimeError(f"traffic/finalize status={status}")

    verdict_path = verifier_dir / f"verdict_{scenario['name']}.json"
    verdict_proc = subprocess.run(
        [
            sys.executable,
            "modules/attestation/verifier_cli/main.py",
            "--transcript",
            str(transcript_path),
            "--traffic-digest",
            str(verifier_dir / "traffic.digest"),
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
    print(
        f"  → verdict: {verdict['verdict']}  ({_signals_summary(verdict)})",
        flush=True,
    )
    return verdict["verdict"], verdict


# --------------- main -----------------------------------------------------


def _print_summary(rows: list[tuple[str, str, str]]) -> bool:
    print("\n=== Summary ===", flush=True)
    width_name = max(len(r[0]) for r in rows)
    width_exp = max(len(r[1]) for r in rows)
    width_act = max(len(r[2]) for r in rows)
    all_pass = True
    for name, expected, actual in rows:
        ok = expected == actual
        all_pass = all_pass and ok
        mark = "OK" if ok else "FAIL"
        print(
            f"  {name:<{width_name}}  expected={expected:<{width_exp}}  "
            f"actual={actual:<{width_act}}  {mark}",
            flush=True,
        )
    print("\nALL PASS" if all_pass else "\nFAILED", flush=True)
    return all_pass


def _run_scenario_local(
    scenario: dict[str, Any],
    *,
    log_root: Path,
    duration_s: float,
    lora_url: str,
) -> tuple[str, dict[str, Any]]:
    """Spawn a fresh prover+verifier for one scenario, run it, tear down.

    Each scenario gets its own pair so traffic.bin / claimed totals reset
    between scenarios — `/traffic/finalize` is one-shot and `recorded_tasks`
    accumulates over a prover's lifetime.
    """
    work = log_root / scenario["name"]
    prover_dir = work / "prover"
    verifier_dir = work / "verifier"
    prover_dir.mkdir(parents=True, exist_ok=True)
    verifier_dir.mkdir(parents=True, exist_ok=True)

    verifier_proc, prover_proc, verifier_port, prover_port = _spawn_servers(
        prover_dir, verifier_dir, log_dir=work
    )
    try:
        _await_health("127.0.0.1", prover_port)
        _await_health("127.0.0.1", verifier_port)
        return _run_scenario(
            scenario,
            prover_url=f"http://127.0.0.1:{prover_port}",
            verifier_url=f"http://127.0.0.1:{verifier_port}",
            verifier_dir=verifier_dir,
            duration_s=duration_s,
            lora_url=lora_url,
        )
    finally:
        _terminate(prover_proc)
        _terminate(verifier_proc)


def main() -> int:
    parser = argparse.ArgumentParser(description="prover-verifier-demo driver")
    parser.add_argument("--per-scenario", type=float, default=5.0, help="seconds per scenario")
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use PROVER_HOST/PROVER_PORT/VERIFIER_HOST/VERIFIER_PORT instead of spawning.",
    )
    args = parser.parse_args()

    print("=== Prover ↔ Verifier demo ===", flush=True)

    log_root = Path(tempfile.mkdtemp(prefix="prover-verifier-demo-"))
    lora_server: _ThreadedServer | None = None
    remote_prover_dir = log_root / "remote-prover"
    remote_verifier_dir = log_root / "remote-verifier"

    try:
        lora_server, lora_url = _start_lora_url_server()

        results: list[tuple[str, str, str]] = []

        if args.remote:
            prover_host = os.environ.get("PROVER_HOST", "127.0.0.1")
            prover_port = int(os.environ.get("PROVER_PORT", "0"))
            verifier_host = os.environ.get("VERIFIER_HOST", "127.0.0.1")
            verifier_port = int(os.environ.get("VERIFIER_PORT", "0"))
            if not (prover_port and verifier_port):
                print(
                    "remote mode requires PROVER_PORT/VERIFIER_PORT to be non-zero",
                    file=sys.stderr,
                )
                return 2
            _await_health(prover_host, prover_port)
            _await_health(verifier_host, verifier_port)
            print(f"Prover @ {prover_host}:{prover_port}", flush=True)
            print(f"Verifier @ {verifier_host}:{verifier_port}", flush=True)
            print("Both healthy.", flush=True)
            remote_prover_dir.mkdir(parents=True, exist_ok=True)
            remote_verifier_dir.mkdir(parents=True, exist_ok=True)
            for scenario in _SCENARIOS:
                actual, _verdict = _run_scenario(
                    scenario,
                    prover_url=f"http://{prover_host}:{prover_port}",
                    verifier_url=f"http://{verifier_host}:{verifier_port}",
                    verifier_dir=remote_verifier_dir,
                    duration_s=args.per_scenario,
                    lora_url=lora_url,
                )
                results.append((scenario["name"], scenario["expected"], actual))
        else:
            print(
                "Spawning a fresh prover+verifier per scenario (loopback).",
                flush=True,
            )
            for scenario in _SCENARIOS:
                actual, _verdict = _run_scenario_local(
                    scenario,
                    log_root=log_root,
                    duration_s=args.per_scenario,
                    lora_url=lora_url,
                )
                results.append((scenario["name"], scenario["expected"], actual))

        all_pass = _print_summary(results)
        print(f"\nLogs: {log_root}", flush=True)
        return 0 if all_pass else 1
    finally:
        if lora_server is not None:
            lora_server.shutdown()
            lora_server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
