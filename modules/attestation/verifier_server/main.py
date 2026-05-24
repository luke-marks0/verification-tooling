#!/usr/bin/env python3
"""Verifier server (prover-verifier-demo).

Stdlib HTTP server that owns the verifier-side endpoints. Phase 3.1 lands
the skeleton + /traffic ingest; subsequent phases add scheduling,
finalize, and verdict logic.

Usage:
    python3 modules/attestation/verifier_server/main.py \\
        --host 127.0.0.1 --port 0 --port-file /tmp/verifier.port \\
        --out-dir /tmp/verifier-demo \\
        --prover-base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.attestation.proverdet.scheduler import HttpProverClient, VerifierScheduler  # noqa: E402
from modules.attestation.proverdet.transcript import TranscriptLog  # noqa: E402


class VerifierState:
    """Mutable state shared across verifier handler threads."""

    def __init__(
        self,
        *,
        out_dir: Path,
        prover_base_url: str,
        seed: int = 0,
        graph_period_ms: int = 1000,
        replay_period_ms: int = 2000,
        autostart_scheduler: bool = True,
    ) -> None:
        self.out_dir = out_dir
        self.prover_base_url = prover_base_url
        self.transcript = TranscriptLog(out_dir / "transcript.jsonl")
        self._traffic_seq = 0
        self._traffic_lock = threading.Lock()

        # Single concatenated traffic file + running sha256.
        self.traffic_path = out_dir / "traffic.bin"
        self.traffic_digest_path = out_dir / "traffic.digest"
        self.traffic_path.write_bytes(b"")  # truncate
        self._hasher = hashlib.sha256()
        self._traffic_size = 0
        self._finalized: dict[str, object] | None = None  # cached on first finalize

        # Scheduler is started by main() after the prover URL is known
        # reachable. Tests with autostart_scheduler=False can drive
        # `run_for_ticks` from the test thread instead.
        self.scheduler = VerifierScheduler(
            client=HttpProverClient(prover_base_url),
            transcript=self.transcript,
            seed=seed,
            graph_period_ms=graph_period_ms,
            replay_period_ms=replay_period_ms,
        )
        self.autostart_scheduler = autostart_scheduler

    def next_traffic_seq(self) -> int:
        with self._traffic_lock:
            self._traffic_seq += 1
            return self._traffic_seq

    def append_traffic(self, data: bytes) -> int:
        """Append bytes to traffic.bin and update running hash.

        Returns the seq for the appended chunk. Raises RuntimeError if
        the stream has already been finalized.
        """
        with self._traffic_lock:
            if self._finalized is not None:
                raise RuntimeError("traffic stream already finalized")
            self._traffic_seq += 1
            with self.traffic_path.open("ab") as f:
                f.write(data)
            self._hasher.update(data)
            self._traffic_size += len(data)
            return self._traffic_seq

    def finalize_traffic(self) -> dict[str, object]:
        """Idempotent: return the cached digest+size on second call."""
        with self._traffic_lock:
            if self._finalized is not None:
                return self._finalized
            digest = "sha256:" + self._hasher.hexdigest()
            self.traffic_digest_path.write_text(digest + "\n", encoding="utf-8")
            self._finalized = {
                "digest": digest,
                "size_bytes": self._traffic_size,
            }
            return self._finalized


class VerifierHandler(BaseHTTPRequestHandler):
    state: VerifierState | None = None

    server_version = "VerifierServer/0.1"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._send_json(200, {"ok": True})
        return self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/traffic":
            return self._handle_post_traffic()
        if self.path == "/traffic/finalize":
            return self._handle_post_traffic_finalize()
        return self._send_json(404, {"error": "not found"})

    def _handle_post_traffic(self) -> None:
        if self.state is None:
            return self._send_json(500, {"error": "verifier state not initialized"})
        body = self._read_body()
        try:
            seq = self.state.append_traffic(body)
        except RuntimeError as exc:
            return self._send_json(409, {"error": str(exc)})
        # Record only the digest in the transcript (full bytes live in
        # traffic.bin). payload_path points at the consolidated file.
        self.state.transcript.record(
            direction="received",
            endpoint="/traffic",
            payload=body,
            payload_path=self.state.traffic_path.name,
        )
        return self._send_json(200, {"received_bytes": len(body), "seq": seq})

    def _handle_post_traffic_finalize(self) -> None:
        if self.state is None:
            return self._send_json(500, {"error": "verifier state not initialized"})
        result = self.state.finalize_traffic()
        # Record the finalize event in the transcript. payload is the
        # digest itself so the verdict engine can find it without a
        # separate file fetch.
        digest = str(result["digest"])
        self.state.transcript.record(
            direction="received",
            endpoint="/traffic/finalize",
            payload=digest.encode("utf-8"),
            payload_path=self.state.traffic_digest_path.name,
        )
        return self._send_json(200, dict(result))


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
    parser = argparse.ArgumentParser(description="Verifier server (prover-verifier-demo)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prover-base-url", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graph-period-ms", type=int, default=1000)
    parser.add_argument("--replay-period-ms", type=int, default=2000)
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Skip starting the active scheduler thread (useful for traffic-only smoke tests).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    state = VerifierState(
        out_dir=args.out_dir,
        prover_base_url=args.prover_base_url,
        seed=args.seed,
        graph_period_ms=args.graph_period_ms,
        replay_period_ms=args.replay_period_ms,
        autostart_scheduler=not args.no_scheduler,
    )
    VerifierHandler.state = state

    server = ThreadedHTTPServer((args.host, args.port), VerifierHandler)
    bound_host, bound_port = server.server_address[0], server.server_address[1]
    if args.port_file:
        _write_port_file(args.port_file, bound_port)

    if state.autostart_scheduler:
        state.scheduler.start()

    print(
        f"verifier: serving on {bound_host}:{bound_port} "
        f"out_dir={args.out_dir} prover_base_url={args.prover_base_url} "
        f"scheduler={'on' if state.autostart_scheduler else 'off'}",
        flush=True,
    )

    def shutdown(signum: int, _frame: Any) -> None:
        print(f"verifier: caught signal {signum}, shutting down", flush=True)
        state.scheduler.stop(timeout=2.0)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        server.serve_forever()
    finally:
        state.scheduler.stop(timeout=2.0)
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
