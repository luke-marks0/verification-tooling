from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from tests.proverdet._helpers import (
    REPO_ROOT,
    read_bound_port,
    sandbox_env,
)


class TestSchedulerAgainstRealProver(unittest.TestCase):
    """Live two-process test: verifier scheduler hits a real prover."""

    prover: subprocess.Popen[bytes] | None = None
    verifier: subprocess.Popen[bytes] | None = None
    tmp: tempfile.TemporaryDirectory[str] | None = None

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.prover_dir = Path(self.tmp.name) / "prover"
        self.verifier_dir = Path(self.tmp.name) / "verifier"
        self.prover_dir.mkdir()
        self.verifier_dir.mkdir()

        prover_port = self.prover_dir / "port"
        verifier_port = self.verifier_dir / "port"

        self.prover = subprocess.Popen(
            [
                sys.executable,
                "cmd/prover/main.py",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--port-file",
                str(prover_port),
                "--run-id",
                "test-run",
                "--out-dir",
                str(self.prover_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(REPO_ROOT),
            env=sandbox_env(),
        )
        try:
            self.prover_port = read_bound_port(prover_port, timeout_s=10.0)
        except Exception:
            self.prover.terminate()
            self.fail("prover never bound")

        prover_url = f"http://127.0.0.1:{self.prover_port}"

        self.verifier = subprocess.Popen(
            [
                sys.executable,
                "cmd/verifier_server/main.py",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--port-file",
                str(verifier_port),
                "--out-dir",
                str(self.verifier_dir),
                "--prover-base-url",
                prover_url,
                "--seed",
                "7",
                "--graph-period-ms",
                "100",
                "--replay-period-ms",
                "200",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(REPO_ROOT),
            env=sandbox_env(),
        )
        try:
            self.verifier_port = read_bound_port(verifier_port, timeout_s=10.0)
        except Exception:
            self.verifier.terminate()
            self.prover.terminate()
            self.fail("verifier never bound")

    def tearDown(self) -> None:
        for p in (self.verifier, self.prover):
            if p is not None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=5)
        if self.tmp is not None:
            self.tmp.cleanup()

    def _read_transcript(self) -> list[dict]:
        path = self.verifier_dir / "transcript.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_scheduler_emits_graph_and_replay_to_real_prover(self) -> None:
        # Wait up to ~5s for the scheduler to issue at least one of each.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            entries = self._read_transcript()
            graph_sent = [
                e for e in entries if e["endpoint"] == "/graph" and e["direction"] == "sent"
            ]
            graph_recv = [
                e for e in entries if e["endpoint"] == "/graph" and e["direction"] == "received"
            ]
            replay_sent = [
                e for e in entries if e["endpoint"] == "/replay" and e["direction"] == "sent"
            ]
            replay_recv = [
                e for e in entries if e["endpoint"] == "/replay" and e["direction"] == "received"
            ]
            if graph_sent and graph_recv and replay_sent and replay_recv:
                break
            time.sleep(0.1)
        else:
            self.fail(f"scheduler did not exchange both /graph and /replay in 5s: {entries!r}")

        # All received entries should have a 200 status (real prover, healthy).
        for e in graph_recv + replay_recv:
            self.assertEqual(e["status_code"], 200)

        # Phase 6.4 DoD: at least one replay verdict landed and it was pass.
        verdicts = [e for e in entries if e["endpoint"].startswith("/replay/verdict/")]
        self.assertGreaterEqual(len(verdicts), 1, "no /replay/verdict entries recorded")
        for v in verdicts:
            self.assertEqual(v["status_code"], 200, f"verdict not pass: {v}")


if __name__ == "__main__":
    unittest.main()
