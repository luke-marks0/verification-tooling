from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from tests.helpers import read_json, run_cmd


@unittest.skipUnless(
    os.getenv("RUNNER_TP_TEST", "").lower() in ("1", "true"),
    "TP determinism test requires multi-GPU and RUNNER_TP_TEST=1",
)
class TestD4TPDeterminism(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch

            self.gpu_count = torch.cuda.device_count()
        except Exception:
            self.gpu_count = 0
        if self.gpu_count < 4:
            self.skipTest(f"Need >= 4 GPUs, found {self.gpu_count}")

        # Pin NCCL for determinism
        os.environ["NCCL_ALGO"] = "Ring"
        os.environ["NCCL_PROTO"] = "Simple"
        os.environ["NCCL_DEBUG"] = "WARN"

    def test_tp4_runner_outputs_are_reproducible(self) -> None:
        manifest = "modules/inference/manifests/qwen2.5-32b-tp4.manifest.json"
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            lock_resolved = tdir / "resolved.lock.json"
            lock_built = tdir / "built.lock.json"
            run_a = tdir / "run-a"
            run_b = tdir / "run-b"
            report = tdir / "verify_report.json"
            summary = tdir / "verify_summary.txt"

            run_cmd(["python3", "modules/inference/resolver/main.py",
                     "--manifest", manifest,
                     "--lockfile-out", str(lock_resolved)])
            run_cmd(["python3", "modules/build/builder/main.py",
                     "--lockfile", str(lock_resolved),
                     "--lockfile-out", str(lock_built)])
            run_cmd(["python3", "modules/inference/runner/main.py",
                     "--manifest", manifest,
                     "--lockfile", str(lock_built),
                     "--out-dir", str(run_a),
                     "--mode", "vllm",
                     "--replica-id", "replica-0"])
            run_cmd(["python3", "modules/inference/runner/main.py",
                     "--manifest", manifest,
                     "--lockfile", str(lock_built),
                     "--out-dir", str(run_b),
                     "--mode", "vllm",
                     "--replica-id", "replica-0"])
            run_cmd([
                "python3", "modules/attestation/verifier/main.py",
                "--baseline", str(run_a / "run_bundle.v1.json"),
                "--candidate", str(run_b / "run_bundle.v1.json"),
                "--report-out", str(report),
                "--summary-out", str(summary),
            ])

            verify = read_json(report)
            self.assertEqual(
                verify["status"], "conformant",
                f"Expected conformant, got {verify['status']}. "
                f"First divergence: {verify.get('first_divergence', 'N/A')}",
            )


if __name__ == "__main__":
    unittest.main()
