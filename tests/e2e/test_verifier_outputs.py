from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import read_json, run_cmd


class TestVerifierOutputs(unittest.TestCase):
    def test_verifier_emits_json_and_text_outputs(self) -> None:
        manifest = "tests/fixtures/positive/manifest.v1.example.json"
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            lock_resolved = tdir / "resolved.lock.json"
            lock_built = tdir / "built.lock.json"
            run_a = tdir / "run-a"
            run_b = tdir / "run-b"
            report = tdir / "verify_report.json"
            summary = tdir / "verify_summary.txt"

            run_cmd(["python3", "modules/inference/resolver/main.py", "--manifest", manifest, "--lockfile-out", str(lock_resolved)])
            run_cmd(["python3", "modules/build/builder/main.py", "--lockfile", str(lock_resolved), "--lockfile-out", str(lock_built)])
            run_cmd(["python3", "modules/inference/runner/main.py", "--manifest", manifest, "--lockfile", str(lock_built), "--out-dir", str(run_a)])
            run_cmd(["python3", "modules/inference/runner/main.py", "--manifest", manifest, "--lockfile", str(lock_built), "--out-dir", str(run_b)])
            run_cmd([
                "python3",
                "modules/attestation/verifier/main.py",
                "--baseline",
                str(run_a / "run_bundle.v1.json"),
                "--candidate",
                str(run_b / "run_bundle.v1.json"),
                "--report-out",
                str(report),
                "--summary-out",
                str(summary),
            ])

            report_data = read_json(report)
            self.assertEqual(report_data["verify_report_version"], "v1")
            self.assertTrue(summary.exists())
            self.assertIn("Status:", summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
