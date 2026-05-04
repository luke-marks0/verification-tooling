from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from pkg.common.contracts import validate_with_schema
from tests.proverdet._helpers import (
    REPO_ROOT,
    http_post_json,
    http_post_ndjson,
    read_bound_port,
    sandbox_env,
)


class _ProverFixture(unittest.TestCase):
    proc: subprocess.Popen[bytes] | None = None
    tmp: tempfile.TemporaryDirectory[str] | None = None
    port: int

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
            if self.proc is not None:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            self.fail("prover never bound")

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


class TestReplayEndpoint(_ProverFixture):
    def _replay_request(self, replay_id: str = "r-1") -> dict:
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
                "rounds": 1,
                "report_every_ms": 100,
            },
            "auxiliary": [],
        }

    def test_post_replay_returns_valid_evidence(self) -> None:
        status, entries = http_post_ndjson(
            f"http://127.0.0.1:{self.port}/replay", self._replay_request()
        )
        self.assertEqual(status, 200)
        self.assertEqual(entries[-1]["kind"], "evidence")
        ev_body = {k: v for k, v in entries[-1].items() if k != "kind"}
        self.assertEqual(ev_body["replay_id"], "r-1")
        validate_with_schema("replay_evidence.v1.schema.json", ev_body)

    def test_post_replay_with_missing_pod_id_returns_400(self) -> None:
        bad = self._replay_request()
        del bad["pod_id"]
        status, body = http_post_json(f"http://127.0.0.1:{self.port}/replay", bad)
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_post_replay_with_invalid_dtype_returns_400(self) -> None:
        bad = self._replay_request()
        bad["proof_of_work"]["dtype"] = "fp64"  # not in {bf16, fp16, int8}
        status, body = http_post_json(f"http://127.0.0.1:{self.port}/replay", bad)
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_get_replay_returns_404(self) -> None:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/replay")
            self.fail("expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_capture_log_records_request_and_response(self) -> None:
        import json as _json

        status, _ = http_post_ndjson(f"http://127.0.0.1:{self.port}/replay", self._replay_request())
        self.assertEqual(status, 200)
        capture = Path(self.tmp.name) / "capture.jsonl"  # type: ignore[arg-type]
        lines = [_json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
        # At minimum: one received request, one sent response on /replay.
        replay_entries = [e for e in lines if e["endpoint"] == "/replay"]
        directions = {e["direction"] for e in replay_entries}
        self.assertIn("sent", directions)
        self.assertIn("received", directions)
        # Seqs are monotonic.
        seqs = [e["seq"] for e in lines]
        self.assertEqual(seqs, sorted(seqs))


if __name__ == "__main__":
    unittest.main()
