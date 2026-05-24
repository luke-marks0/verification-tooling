from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from tests.proverdet._helpers import (
    REPO_ROOT,
    http_get_json,
    http_post_bytes,
    read_bound_port,
    sandbox_env,
)


class _VerifierFixture(unittest.TestCase):
    proc: subprocess.Popen[bytes] | None = None
    tmp: tempfile.TemporaryDirectory[str] | None = None
    port: int

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        port_file = Path(self.tmp.name) / "bound.port"
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "modules/attestation/verifier_server/main.py",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--port-file",
                str(port_file),
                "--out-dir",
                self.tmp.name,
                "--prover-base-url",
                "http://127.0.0.1:1",  # unused this test (scheduler off)
                "--no-scheduler",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(REPO_ROOT),
            env=sandbox_env(),
        )
        try:
            self.port = read_bound_port(port_file, timeout_s=10.0)
        except Exception:
            if self.proc is not None:
                self.proc.terminate()
                self.proc.wait(timeout=5)
                stderr = (self.proc.stderr.read() if self.proc.stderr else b"").decode(
                    errors="replace"
                )
                self.fail(f"verifier never bound. stderr=\n{stderr}")

    def tearDown(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        if self.tmp is not None:
            self.tmp.cleanup()


class TestVerifierLifecycle(_VerifierFixture):
    def test_health_returns_ok(self) -> None:
        status, body = http_get_json(f"http://127.0.0.1:{self.port}/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"ok": True})

    def test_unknown_endpoint_returns_404(self) -> None:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/no-such")
            self.fail("expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)


class TestTrafficIngest(_VerifierFixture):
    def _read_transcript_lines(self) -> list[dict]:
        path = Path(self.tmp.name) / "transcript.jsonl"  # type: ignore[arg-type]
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_post_traffic_persists_bytes_and_logs_entry(self) -> None:
        payload = b"a" * 1024
        status, _ = http_post_bytes(
            f"http://127.0.0.1:{self.port}/traffic",
            payload,
        )
        self.assertEqual(status, 200)

        out_dir = Path(self.tmp.name)  # type: ignore[arg-type]
        bin_path = out_dir / "traffic.bin"
        self.assertTrue(bin_path.exists())
        self.assertEqual(bin_path.read_bytes(), payload)

        entries = self._read_transcript_lines()
        traffic_entries = [e for e in entries if e["endpoint"] == "/traffic"]
        self.assertEqual(len(traffic_entries), 1)
        self.assertEqual(traffic_entries[0]["direction"], "received")

    def test_post_traffic_records_received_direction(self) -> None:
        http_post_bytes(f"http://127.0.0.1:{self.port}/traffic", b"hello")
        entries = self._read_transcript_lines()
        traffic_entries = [e for e in entries if e["endpoint"] == "/traffic"]
        self.assertEqual(traffic_entries[0]["direction"], "received")

    def test_post_traffic_increments_seq_per_chunk(self) -> None:
        http_post_bytes(f"http://127.0.0.1:{self.port}/traffic", b"a")
        http_post_bytes(f"http://127.0.0.1:{self.port}/traffic", b"b")
        http_post_bytes(f"http://127.0.0.1:{self.port}/traffic", b"c")
        entries = self._read_transcript_lines()
        traffic = [e for e in entries if e["endpoint"] == "/traffic"]
        self.assertEqual(len(traffic), 3)
        seqs = [e["seq"] for e in traffic]
        self.assertEqual(seqs, sorted(seqs))


if __name__ == "__main__":
    unittest.main()
