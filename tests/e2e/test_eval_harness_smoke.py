"""Smoke test for the eval sweep harness (Task 9.1).

Runs `experiments/prover-verifier-demo/scripts/run_eval.py --smoke` against
a temporary output dir and checks the JSONL row schema. Doesn't pin
exact verdicts — Phase 9 plotting/HTML viewer assert that — only that
each row has the required fields and uses the binary verdict vocabulary.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.proverdet._helpers import REPO_ROOT, sandbox_env


REQUIRED_FIELDS = {
    "workload",
    "knob_value",
    "verdict",
    "signals",
    "observed_flops",
    "traffic_size",
}
ALLOWED_VERDICTS = {"inference", "training_or_exfil", "unknown"}


class TestEvalHarnessSmoke(unittest.TestCase):
    def test_smoke_run_writes_one_row_per_workload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "eval"
            result = subprocess.run(
                [
                    sys.executable,
                    "experiments/prover-verifier-demo/scripts/run_eval.py",
                    "--smoke",
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=str(REPO_ROOT),
                env=sandbox_env(),
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout={result.stdout}\nstderr={result.stderr}",
            )
            results_path = out_dir / "results.jsonl"
            self.assertTrue(results_path.exists(), "results.jsonl not written")
            rows = [
                json.loads(line)
                for line in results_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(rows), 3, rows)
            seen_workloads = set()
            for row in rows:
                missing = REQUIRED_FIELDS - row.keys()
                self.assertFalse(missing, f"row {row!r} missing fields: {missing}")
                self.assertIn(row["verdict"], ALLOWED_VERDICTS, row)
                self.assertIsInstance(row["signals"], dict, row)
                self.assertIsInstance(row["observed_flops"], int, row)
                self.assertIsInstance(row["traffic_size"], int, row)
                seen_workloads.add(row["workload"])
            self.assertEqual(
                seen_workloads, {"benign", "mixed_lora", "lora_loading"}, seen_workloads
            )


if __name__ == "__main__":
    unittest.main()
