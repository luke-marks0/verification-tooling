"""End-to-end adversarial workload test (mixed_lora).

Boots prover + verifier, runs the mixed_lora workload (gradient_steps=4)
through the wire protocol, finalizes traffic, and runs the verdict CLI.
Until Phase 8.3 the verdict is "unknown" — that's intentional. The 8.3
commit will edit this test in place to assert "training_or_exfil".
"""

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
    http_post_json,
    read_bound_port,
    sandbox_env,
)


class TestAdversarialE2E(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.prover_dir = Path(self.tmp.name) / "prover"
        self.verifier_dir = Path(self.tmp.name) / "verifier"
        self.prover_dir.mkdir()
        self.verifier_dir.mkdir()

        verifier_port_file = self.verifier_dir / "port"
        prover_port_file = self.prover_dir / "port"

        # Verifier first; prover needs its URL. --no-scheduler keeps the
        # transcript focused on traffic + finalize for this test (the
        # scheduler's /graph and /replay flow already lives in
        # test_verifier_scheduler_against_prover.py).
        self.verifier = subprocess.Popen(
            [
                sys.executable,
                "cmd/verifier_server/main.py",
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
                "cmd/prover/main.py",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--port-file",
                str(prover_port_file),
                "--run-id",
                "e2e-adv",
                "--out-dir",
                str(self.prover_dir),
                "--verifier-url",
                verifier_url,
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

    def test_mixed_lora_workload_runs_and_emits_verdict(self) -> None:
        # 1. Start mixed_lora with gradient_steps=4 — the cheating knob.
        params = {
            "prompts": ["adv-1", "adv-2"],
            "use_vllm": False,
            "seed": 11,
            "gradient_steps": 4,
            "matmul_dim": 8,  # tiny to keep CPU runtime bounded
        }
        status, body = http_post_json(
            f"http://127.0.0.1:{self.prover_port}/workload/start",
            {"name": "mixed_lora", "params": params},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["started"], "mixed_lora")

        # 2. Let it run; mixed_lora is fast on CPU so 0.5s is plenty.
        time.sleep(0.5)

        # 3. Stop + finalize.
        status, _ = http_post_json(f"http://127.0.0.1:{self.prover_port}/workload/stop", {})
        self.assertEqual(status, 200)
        status, fbody = http_post_json(
            f"http://127.0.0.1:{self.verifier_port}/traffic/finalize", {}
        )
        self.assertEqual(status, 200)
        # Inference traffic only (gradient steps emit no frames):
        # 2 prompts * 10 frames * 256 bytes = 5120 bytes.
        self.assertEqual(fbody["size_bytes"], 2 * 10 * 256)

        # 4. Run the verdict CLI.
        verdict_path = self.verifier_dir / "verdict.json"
        result = subprocess.run(
            [
                sys.executable,
                "cmd/verifier_cli/main.py",
                "--transcript",
                str(self.verifier_dir / "transcript.jsonl"),
                "--out",
                str(verdict_path),
            ],
            cwd=str(REPO_ROOT),
            env=sandbox_env(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, f"stderr={result.stderr}")

        # 5. Until Phase 8.3 lands the verdict is "unknown" — that's fine.
        # Phase 8.3 will edit this assertion to "training_or_exfil".
        self.assertTrue(verdict_path.exists())
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        self.assertIn("verdict", verdict)
        self.assertIn("reasons", verdict)


if __name__ == "__main__":
    unittest.main()
