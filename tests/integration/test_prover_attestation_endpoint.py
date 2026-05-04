from __future__ import annotations

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
    http_post_ndjson,
    read_bound_port,
    sandbox_env,
)


class TestAttestationEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        port_file = Path(self.tmp.name) / "bound.port"
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "cmd/prover/main.py",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--port-file",
                str(port_file),
                "--run-id",
                "test-run",
                "--out-dir",
                self.tmp.name,
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
            self.fail("prover never bound")

    def tearDown(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
        self.tmp.cleanup()

    def _replay_request(self, replay_id: str = "att-r-1") -> dict[str, object]:
        return {
            "replay_id": replay_id,
            "pod_id": "pod-a",
            "target": {"kind": "task", "task_id": "task-0"},
            "erasure": {
                "challenge_seed": "deadbeef",
                "deadline_ms": 1000,
                "rounds": 2,
            },
            "proof_of_work": {
                "matmul_dim": 8,
                "dtype": "int8",
                "rounds": 2,
                "report_every_ms": 100,
            },
            "auxiliary": [],
        }

    def test_get_attestation_returns_stored_body(self) -> None:
        status, entries = http_post_ndjson(
            f"http://127.0.0.1:{self.port}/replay", self._replay_request()
        )
        self.assertEqual(status, 200)
        pow_entries = [e for e in entries if e["kind"] == "pow"]
        self.assertGreater(len(pow_entries), 0)
        att_id = pow_entries[0]["freivalds_attestation_id"]

        status, att = http_get_json(f"http://127.0.0.1:{self.port}/attestation/{att_id}")
        self.assertEqual(status, 200)
        self.assertIn("challenge", att)
        self.assertIn("response", att)
        self.assertIn("matmul_id", att)

    def test_get_unknown_attestation_returns_404(self) -> None:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/attestation/no-such-id")
            self.fail("expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)


if __name__ == "__main__":
    unittest.main()
