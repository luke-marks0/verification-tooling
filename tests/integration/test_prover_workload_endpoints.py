from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from tests.proverdet._helpers import (
    REPO_ROOT,
    http_post_json,
    read_bound_port,
    sandbox_env,
)


class TestWorkloadEndpoints(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.prover_dir = Path(self.tmp.name) / "prover"
        self.verifier_dir = Path(self.tmp.name) / "verifier"
        self.prover_dir.mkdir()
        self.verifier_dir.mkdir()

        verifier_port_file = self.verifier_dir / "port"
        prover_port_file = self.prover_dir / "port"

        self.verifier = subprocess.Popen(
            [
                sys.executable,
                "modules/attestation/verifier_server/main.py",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--port-file",
                str(verifier_port_file),
                "--out-dir",
                str(self.verifier_dir),
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
            self.verifier_port = read_bound_port(verifier_port_file, timeout_s=10.0)
        except Exception:
            self.verifier.terminate()
            self.fail("verifier never bound")

        verifier_url = f"http://127.0.0.1:{self.verifier_port}"
        self.prover = subprocess.Popen(
            [
                sys.executable,
                "modules/attestation/prover/main.py",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--port-file",
                str(prover_port_file),
                "--run-id",
                "wl-test",
                "--out-dir",
                str(self.prover_dir),
                "--verifier-url",
                verifier_url,
                "--debug-mode",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(REPO_ROOT),
            env=sandbox_env(),
        )
        try:
            self.prover_port = read_bound_port(prover_port_file, timeout_s=10.0)
        except Exception:
            self.prover.terminate()
            self.verifier.terminate()
            self.fail("prover never bound")

    def tearDown(self) -> None:
        for p in (self.prover, self.verifier):
            if p is not None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=5)
        self.tmp.cleanup()

    def test_starting_benign_publishes_traffic(self) -> None:
        status, body = http_post_json(
            f"http://127.0.0.1:{self.prover_port}/workload/start",
            {"name": "benign", "params": {"prompts": ["hi", "ho"], "use_vllm": False, "seed": 1}},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["started"], "benign")

        # Workload finishes near-instantly for the synthetic path.
        time.sleep(0.5)
        status, _ = http_post_json(f"http://127.0.0.1:{self.prover_port}/workload/stop", {})
        self.assertEqual(status, 200)

        status, fbody = http_post_json(
            f"http://127.0.0.1:{self.verifier_port}/traffic/finalize", {}
        )
        self.assertEqual(status, 200)
        # 2 prompts * 10 frames * 256 bytes = 5120
        self.assertEqual(fbody["size_bytes"], 2 * 10 * 256)

    def test_starting_second_workload_returns_409(self) -> None:
        # delay_per_prompt_s keeps the workload alive long enough for the
        # second start to land while the first is still running.
        params = {
            "prompts": ["a", "b", "c"],
            "use_vllm": False,
            "seed": 7,
            "delay_per_prompt_s": 0.5,
        }
        http_post_json(
            f"http://127.0.0.1:{self.prover_port}/workload/start",
            {"name": "benign", "params": params},
        )
        time.sleep(0.05)  # let the workload thread enter its loop
        status, body = http_post_json(
            f"http://127.0.0.1:{self.prover_port}/workload/start",
            {"name": "benign", "params": params},
        )
        self.assertEqual(status, 409)
        self.assertIn("error", body)
        http_post_json(f"http://127.0.0.1:{self.prover_port}/workload/stop", {})

    def test_unknown_workload_name_returns_404(self) -> None:
        status, body = http_post_json(
            f"http://127.0.0.1:{self.prover_port}/workload/start",
            {"name": "no-such-workload", "params": {}},
        )
        self.assertEqual(status, 404)
        self.assertIn("error", body)

    def test_missing_name_returns_400(self) -> None:
        status, _body = http_post_json(
            f"http://127.0.0.1:{self.prover_port}/workload/start",
            {"params": {}},
        )
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
