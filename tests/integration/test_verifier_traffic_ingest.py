from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.proverdet._helpers import (
    REPO_ROOT,
    http_post_bytes,
    http_post_json,
    read_bound_port,
    sandbox_env,
)


class _VerifierFixture(unittest.TestCase):
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
                "http://127.0.0.1:1",
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
            self.proc.terminate()
            self.fail("verifier never bound")

    def tearDown(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.tmp.cleanup()


class TestTrafficIngestFinalize(_VerifierFixture):
    def test_streams_concatenate_into_traffic_bin(self) -> None:
        chunks = [b"hello ", b"world ", b"!" * 100]
        for c in chunks:
            http_post_bytes(f"http://127.0.0.1:{self.port}/traffic", c)

        out_dir = Path(self.tmp.name)
        bin_path = out_dir / "traffic.bin"
        self.assertTrue(bin_path.exists())
        self.assertEqual(bin_path.read_bytes(), b"".join(chunks))

    def test_finalize_writes_digest_matching_manual_hash(self) -> None:
        chunks = [b"a" * 1000, b"b" * 500, b"c" * 250]
        for c in chunks:
            http_post_bytes(f"http://127.0.0.1:{self.port}/traffic", c)
        status, body = http_post_json(f"http://127.0.0.1:{self.port}/traffic/finalize", {})
        self.assertEqual(status, 200)

        bin_path = Path(self.tmp.name) / "traffic.bin"
        digest_path = Path(self.tmp.name) / "traffic.digest"
        self.assertTrue(digest_path.exists())
        expected = "sha256:" + hashlib.sha256(b"".join(chunks)).hexdigest()
        self.assertEqual(digest_path.read_text().strip(), expected)
        self.assertEqual(body["digest"], expected)
        self.assertEqual(body["size_bytes"], bin_path.stat().st_size)

    def test_finalize_is_idempotent(self) -> None:
        http_post_bytes(f"http://127.0.0.1:{self.port}/traffic", b"abc")
        status1, body1 = http_post_json(f"http://127.0.0.1:{self.port}/traffic/finalize", {})
        status2, body2 = http_post_json(f"http://127.0.0.1:{self.port}/traffic/finalize", {})
        self.assertEqual(status1, 200)
        self.assertEqual(status2, 200)
        self.assertEqual(body1["digest"], body2["digest"])
        self.assertEqual(body1["size_bytes"], body2["size_bytes"])

    def test_traffic_after_finalize_returns_409(self) -> None:
        import urllib.error
        import urllib.request

        http_post_bytes(f"http://127.0.0.1:{self.port}/traffic", b"abc")
        http_post_json(f"http://127.0.0.1:{self.port}/traffic/finalize", {})
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"http://127.0.0.1:{self.port}/traffic",
                    data=b"def",
                    headers={"Content-Type": "application/octet-stream"},
                    method="POST",
                )
            )
            self.fail("expected 409 after finalize")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 409)

    def test_transcript_records_finalize(self) -> None:
        http_post_bytes(f"http://127.0.0.1:{self.port}/traffic", b"x")
        http_post_json(f"http://127.0.0.1:{self.port}/traffic/finalize", {})
        path = Path(self.tmp.name) / "transcript.jsonl"
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        endpoints = {e["endpoint"] for e in entries}
        self.assertIn("/traffic/finalize", endpoints)


if __name__ == "__main__":
    unittest.main()
