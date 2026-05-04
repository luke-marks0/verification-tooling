#!/usr/bin/env python3
"""Prover server (prover-verifier-demo).

Stdlib HTTP server that owns the prover-side endpoints of the prover ↔
verifier protocol. Phase 2 lands the skeleton + /health; subsequent phases
add /graph, /replay, /workload/{start,stop}, /attestation/{id}, and an
optional /debug/emit-frames.

Usage:
    python3 cmd/prover/main.py \\
        --host 127.0.0.1 --port 0 --port-file /tmp/prover.port \\
        --run-id demo-001 --out-dir /tmp/prover-demo
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydantic import ValidationError as PydanticValidationError  # noqa: E402

from pkg.common.contracts import ValidationError, validate_with_schema  # noqa: E402
from pkg.freivalds.backends.stdlib import StdlibBackend  # noqa: E402
from pkg.proverdet.attestation_store import AttestationStore  # noqa: E402
from pkg.proverdet.capture import ProverCaptureLog  # noqa: E402
from pkg.proverdet.graph_builder import build_empty_graph  # noqa: E402
from pkg.proverdet.replay import check_supported, produce_evidence_stream  # noqa: E402
from pkg.proverdet.traffic_publisher import TrafficPublisher  # noqa: E402
from pkg.proverdet.wire import ReplayRequest  # noqa: E402
from pkg.proverdet.workload_runner import WorkloadRunner  # noqa: E402


class ProverState:
    """Shared mutable state for the prover server.

    Mirrors `cmd/server/main.py`'s `ServerState` shape but only owns what the
    prover actually needs: a run id, an output dir, a cross-handler lock.
    Workload thread, capture log, etc. land in later tasks.
    """

    def __init__(
        self,
        *,
        run_id: str,
        out_dir: Path,
        verifier_url: str | None = None,
        debug_mode: bool = False,
    ) -> None:
        self.run_id = run_id
        self.out_dir = out_dir
        self.verifier_url = verifier_url
        self.debug_mode = debug_mode
        self.lock = threading.Lock()
        self.capture_log = ProverCaptureLog(out_dir / "capture.jsonl")
        self.traffic_publisher: TrafficPublisher | None = (
            TrafficPublisher(verifier_url=verifier_url) if verifier_url else None
        )
        self.recorded_tasks: list[dict[str, object]] = []
        self._tasks_lock = threading.Lock()
        self.workload_runner = WorkloadRunner(
            publish_frame=self._publish_frame,
            record_task=self._record_task,
        )
        self.attestation_store = AttestationStore()
        # Stdlib backend keeps tests CPU-only. The torch backend can swap in
        # for fp16/bf16 paths once GPU is available; the wire dtype check in
        # produce_evidence will raise if the backend can't handle it.
        self.freivalds_backend = StdlibBackend()

    def _publish_frame(self, frame: bytes) -> None:
        if self.traffic_publisher is None:
            return
        self.traffic_publisher.start()
        self.traffic_publisher.publish(frame)

    def _record_task(self, task: dict[str, object]) -> None:
        with self._tasks_lock:
            self.recorded_tasks.append(task)

    def task_totals(self) -> tuple[int, int]:
        """Snapshot (claimed_flops_total, task_count) of recorded tasks."""
        with self._tasks_lock:
            claimed = 0
            for t in self.recorded_tasks:
                v = t.get("claimed_flops", 0)
                if isinstance(v, int):
                    claimed += v
            return claimed, len(self.recorded_tasks)

    def stop(self) -> None:
        try:
            self.workload_runner.stop(timeout=2.0)
        finally:
            if self.traffic_publisher is not None:
                self.traffic_publisher.stop()


class ProverHandler(BaseHTTPRequestHandler):
    """Stdlib request handler for the prover. One method per route family."""

    state: ProverState | None = None

    server_version = "ProverServer/0.1"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        # The default logger writes to stderr at every request; we want
        # the demo to be quiet. The capture log (Task 2.4) is the real audit
        # trail.
        return

    # -- helpers --

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        if self.state is not None:
            self.state.capture_log.record(
                direction="sent",
                endpoint=self.path,
                payload=data,
                status_code=status,
            )

    # -- GET --

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._send_json(200, {"ok": True})
        if self.path == "/graph":
            return self._handle_get_graph()
        if self.path.startswith("/attestation/"):
            return self._handle_get_attestation(self.path[len("/attestation/") :])
        return self._send_json(404, {"error": "not found"})

    def _handle_get_attestation(self, attestation_id: str) -> None:
        if self.state is None:
            return self._send_json(500, {"error": "prover state not initialized"})
        body = self.state.attestation_store.get(attestation_id)
        if body is None:
            return self._send_json(404, {"error": f"unknown attestation: {attestation_id}"})
        return self._send_json(200, body)

    def _handle_get_graph(self) -> None:
        if self.state is None:
            return self._send_json(500, {"error": "prover state not initialized"})
        graph = build_empty_graph(run_id=self.state.run_id)
        body = graph.model_dump(exclude_none=True)
        try:
            validate_with_schema("prover_graph.v1.schema.json", body)
        except ValidationError as exc:
            # Schema mismatch is a programmer error; surface 500 with the
            # message so it shows up in tests.
            return self._send_json(500, {"error": f"graph schema mismatch: {exc}"})
        return self._send_json(200, body)

    # -- POST --

    def do_POST(self) -> None:
        if self.path == "/replay":
            return self._handle_post_replay()
        if self.path == "/workload/start":
            return self._handle_post_workload_start()
        if self.path == "/workload/stop":
            return self._handle_post_workload_stop()
        if self.path == "/debug/emit-frames":
            return self._handle_post_debug_emit_frames()
        return self._send_json(404, {"error": "not found"})

    def _handle_post_workload_start(self) -> None:
        if self.state is None:
            return self._send_json(500, {"error": "prover state not initialized"})
        raw = self._read_body()
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            return self._send_json(400, {"error": f"invalid JSON: {exc}"})
        name = payload.get("name")
        if not isinstance(name, str):
            return self._send_json(400, {"error": "missing or non-string 'name'"})
        params = payload.get("params", {})
        if not isinstance(params, dict):
            return self._send_json(400, {"error": "'params' must be an object"})
        try:
            self.state.workload_runner.start(name=name, params=params)
        except RuntimeError as exc:
            return self._send_json(409, {"error": str(exc)})
        except KeyError as exc:
            return self._send_json(404, {"error": str(exc)})
        except (TypeError, ValueError) as exc:
            return self._send_json(400, {"error": str(exc)})
        return self._send_json(200, {"started": name})

    def _handle_post_workload_stop(self) -> None:
        if self.state is None:
            return self._send_json(500, {"error": "prover state not initialized"})
        was_running = self.state.workload_runner.is_running
        observed_flops = self.state.workload_runner.stop(timeout=10.0)
        claimed_flops, task_count = self.state.task_totals()
        return self._send_json(
            200,
            {
                "stopped": was_running,
                "claimed_flops_total": claimed_flops,
                "observed_flops_total": observed_flops,
                "task_count": task_count,
            },
        )

    def _handle_post_debug_emit_frames(self) -> None:
        """Debug-only: synthesize N deterministic frames and publish them.

        Body: {"count": int, "size_bytes": int, "seed": int (optional)}.
        Frames are sha256(seed || index)[:size_bytes] tiled, so the test can
        recompute the same bytes the verifier should receive.
        """
        if self.state is None:
            return self._send_json(500, {"error": "prover state not initialized"})
        if not self.state.debug_mode:
            return self._send_json(404, {"error": "not found"})
        if self.state.traffic_publisher is None:
            return self._send_json(
                500, {"error": "no verifier_url configured for traffic publisher"}
            )
        raw = self._read_body()
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            return self._send_json(400, {"error": f"invalid JSON: {exc}"})
        count = int(payload.get("count", 1))
        size_bytes = int(payload.get("size_bytes", 64))
        seed = int(payload.get("seed", 0))

        # Lazily start the publisher.
        self.state.traffic_publisher.start()
        for i in range(count):
            self.state.traffic_publisher.publish(_synth_frame(seed, i, size_bytes))
        return self._send_json(200, {"published": count, "size_bytes": size_bytes})

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _handle_post_replay(self) -> None:
        raw = self._read_body()
        if self.state is not None:
            self.state.capture_log.record(
                direction="received",
                endpoint=self.path,
                payload=raw,
            )
        if not raw:
            return self._send_json(400, {"error": "empty request body"})
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            return self._send_json(400, {"error": f"invalid JSON: {exc}"})

        # Schema-validate first (fast 400 with a path), then Pydantic-validate
        # (which gives us a typed object).
        try:
            validate_with_schema("replay_request.v1.schema.json", payload)
        except ValidationError as exc:
            return self._send_json(400, {"error": str(exc)})

        try:
            req = ReplayRequest.model_validate(payload)
        except PydanticValidationError as exc:
            return self._send_json(400, {"error": str(exc)})

        if self.state is None:
            return self._send_json(500, {"error": "prover state not initialized"})

        # Pre-flight: surface a 4xx synchronously rather than mid-stream.
        try:
            check_supported(req)
        except ValueError as exc:
            return self._send_json(400, {"error": str(exc)})

        # Stream NDJSON: one application/x-ndjson line per chunk. We
        # don't set Content-Length and rely on Connection: close so the
        # client reads to EOF — simplest path with stdlib's
        # BaseHTTPRequestHandler.
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        captured = bytearray()
        evidence_seen = False
        for chunk in produce_evidence_stream(
            req,
            freivalds_backend=self.state.freivalds_backend,
            attestation_store=self.state.attestation_store,
            erasure_log_dir=self.state.out_dir / "erasure",
        ):
            if chunk["kind"] == "evidence":
                evidence_body = {k: v for k, v in chunk.items() if k != "kind"}
                # Defensive: belt-and-braces validate evidence before sending.
                try:
                    validate_with_schema("replay_evidence.v1.schema.json", evidence_body)
                except ValidationError as exc:
                    err = (
                        json.dumps(
                            {"kind": "error", "error": f"evidence schema mismatch: {exc}"}
                        ).encode("utf-8")
                        + b"\n"
                    )
                    captured.extend(err)
                    self.wfile.write(err)
                    self.wfile.flush()
                    break
                evidence_seen = True
            line = json.dumps(chunk).encode("utf-8") + b"\n"
            captured.extend(line)
            self.wfile.write(line)
            self.wfile.flush()
        self.state.capture_log.record(
            direction="sent",
            endpoint=self.path,
            payload=bytes(captured),
            status_code=200 if evidence_seen else 500,
        )


def _synth_frame(seed: int, index: int, size_bytes: int) -> bytes:
    """Deterministic synthetic frame for the /debug/emit-frames endpoint.

    Tile sha256(seed||index) up to size_bytes. Independent of host —
    same (seed, index, size) always yields the same bytes, so tests can
    recompute the expected concatenated stream.
    """
    import hashlib

    seed_bytes = seed.to_bytes(8, "big", signed=False)
    idx_bytes = index.to_bytes(8, "big", signed=False)
    digest = hashlib.sha256(seed_bytes + idx_bytes).digest()  # 32 bytes
    out = bytearray()
    while len(out) < size_bytes:
        out.extend(digest)
    return bytes(out[:size_bytes])


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _write_port_file(port_file: Path, port: int) -> None:
    port_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(port_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        os.write(fd, f"{port}\n".encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prover server (prover-verifier-demo)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--port-file",
        type=Path,
        default=None,
        help="Write the bound port to this file, fsync, then serve.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--verifier-url",
        default=None,
        help="Base URL of the verifier server (used by traffic publisher).",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Enable test-only endpoints like /debug/emit-frames.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    state = ProverState(
        run_id=args.run_id,
        out_dir=args.out_dir,
        verifier_url=args.verifier_url,
        debug_mode=args.debug_mode,
    )
    ProverHandler.state = state

    server = ThreadedHTTPServer((args.host, args.port), ProverHandler)
    bound_host, bound_port = server.server_address[0], server.server_address[1]

    if args.port_file:
        _write_port_file(args.port_file, bound_port)

    print(
        f"prover: serving on {bound_host}:{bound_port} run_id={args.run_id} out_dir={args.out_dir}",
        flush=True,
    )

    def shutdown(signum: int, _frame: Any) -> None:
        print(f"prover: caught signal {signum}, shutting down", flush=True)
        state.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        server.serve_forever()
    finally:
        state.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
