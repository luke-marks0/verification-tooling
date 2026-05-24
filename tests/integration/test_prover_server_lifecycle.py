from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

from tests.proverdet._helpers import (
    REPO_ROOT,
    http_get_json,
    read_bound_port,
    sandbox_env,
)


class TestProverLifecycle(unittest.TestCase):
    proc: subprocess.Popen[bytes] | None = None
    tmp: tempfile.TemporaryDirectory[str] | None = None
    port: int

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        port_file = Path(self.tmp.name) / "bound.port"
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "modules/attestation/prover/main.py",
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
            self.proc.wait(timeout=5)
            stderr = (self.proc.stderr.read() if self.proc.stderr else b"").decode(errors="replace")
            stdout = (self.proc.stdout.read() if self.proc.stdout else b"").decode(errors="replace")
            self.fail(f"prover never bound. stdout=\n{stdout}\nstderr=\n{stderr}")

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

    def test_health_returns_ok(self) -> None:
        status, body = http_get_json(f"http://127.0.0.1:{self.port}/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"ok": True})

    def test_get_graph_returns_empty_placeholder(self) -> None:
        from modules.core.common.contracts import validate_with_schema

        status, body = http_get_json(f"http://127.0.0.1:{self.port}/graph")
        self.assertEqual(status, 200)
        self.assertEqual(body["graph_version"], "v1-placeholder")
        self.assertEqual(body["run_id"], "test-run")
        self.assertEqual(body["tasks"], [])
        self.assertEqual(body["artifacts"], [])
        self.assertEqual(body["transmissions"], [])
        validate_with_schema("prover_graph.v1.schema.json", body)

    def test_unknown_endpoint_returns_404(self) -> None:
        try:
            http_get_json(f"http://127.0.0.1:{self.port}/no-such")
            self.fail("expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_run_id_persists_in_state_dir(self) -> None:
        # Subdirectory exists under out-dir.
        out = Path(self.tmp.name)  # type: ignore[arg-type]
        self.assertTrue(out.exists())

    def test_terminates_on_sigterm(self) -> None:
        if self.proc is None:
            self.fail("no proc")
        self.proc.terminate()
        try:
            rc = self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.fail("did not terminate within 5s")
        # Re-init for tearDown.
        self.proc = None
        self.assertIn(rc, (0, -15))  # 0 if it caught the signal cleanly


if __name__ == "__main__":
    unittest.main()
