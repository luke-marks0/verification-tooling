from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

from pkg.common.contracts import validate_with_schema
from tests.proverdet._helpers import (
    REPO_ROOT,
    read_bound_port,
    sandbox_env,
)


class TestReplayStreaming(unittest.TestCase):
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
                "stream-test",
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

    def _replay_request(self, replay_id: str = "r-stream") -> dict[str, object]:
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
                "rounds": 3,
                "report_every_ms": 100,
            },
            "auxiliary": [],
        }

    def _post_ndjson(self, payload: dict[str, object]) -> tuple[int, list[dict[str, object]]]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/replay",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10.0) as r:
            entries: list[dict[str, object]] = []
            for raw_line in r:
                line = raw_line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
            return r.status, entries

    def test_replay_streams_pow_then_evidence(self) -> None:
        t0 = time.monotonic()
        status, entries = self._post_ndjson(self._replay_request())
        wall_ms = (time.monotonic() - t0) * 1000.0

        self.assertEqual(status, 200)

        kinds = [e["kind"] for e in entries]
        # exactly 3 pow lines, then one evidence line, in that order.
        pow_count = kinds.count("pow")
        ev_count = kinds.count("evidence")
        self.assertEqual(pow_count, 3)
        self.assertEqual(ev_count, 1)
        self.assertEqual(kinds[-1], "evidence")
        # Every pow line precedes the final evidence.
        first_evidence_idx = kinds.index("evidence")
        self.assertEqual(first_evidence_idx, len(kinds) - 1)

        # Cadence sanity: rounds=3, report_every_ms=100 → ≤ 1.5 * 300 ms.
        # We don't assert a lower bound — small matmuls finish in microseconds.
        self.assertLess(wall_ms, 1.5 * 300 + 1000)  # 1s slack for spawn overhead

        # Final evidence chunk is a full schema-valid ReplayEvidence.
        evidence_chunk = entries[-1]
        ev_body = {k: v for k, v in evidence_chunk.items() if k != "kind"}
        validate_with_schema("replay_evidence.v1.schema.json", ev_body)
        self.assertEqual(ev_body["replay_id"], "r-stream")

    def test_invalid_dtype_returns_400_without_streaming(self) -> None:
        # Pre-stream validation: a dtype that the prover backend can't handle
        # should fail synchronously, before the stream opens.
        bad = self._replay_request()
        bad["proof_of_work"]["dtype"] = "fp64"  # not in {bf16, fp16, int8}
        data = json.dumps(bad).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/replay",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=10.0)
            self.fail("expected 400")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)


if __name__ == "__main__":
    unittest.main()
