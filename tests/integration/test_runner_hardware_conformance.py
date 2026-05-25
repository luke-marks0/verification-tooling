from __future__ import annotations

import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.helpers import read_json, run_cmd, write_json


class TestRunnerHardwareConformance(unittest.TestCase):
    def test_strict_hardware_refuses_nonconforming_runtime(self) -> None:
        manifest = read_json(Path("tests/fixtures/positive/manifest.v1.example.json"))
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            manifest_path = tdir / "manifest.strict.json"
            write_json(manifest_path, manifest)

            observed = copy.deepcopy(manifest["hardware_profile"])
            observed["gpu"]["model"] = "H100-PCIe-80GB"
            runtime_hw = tdir / "runtime.hardware.json"
            write_json(runtime_hw, observed)

            resolved = tdir / "resolved.lock.json"
            built = tdir / "built.lock.json"
            run_cmd([sys.executable, "modules/inference/resolver/main.py", "--manifest", str(manifest_path), "--lockfile-out", str(resolved)])
            run_cmd([sys.executable, "modules/build/builder/main.py", "--lockfile", str(resolved), "--lockfile-out", str(built)])

            proc = subprocess.run(
                [
                    sys.executable,
                    "modules/inference/runner/main.py", "--mode", "mock",
                    "--manifest",
                    str(manifest_path),
                    "--lockfile",
                    str(built),
                    "--out-dir",
                    str(tdir / "run"),
                    "--runtime-hardware",
                    str(runtime_hw),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
            self.assertIn("strict_hardware=true", combined)

    def test_nonstrict_hardware_labels_nonconformant_and_verifier_reports_it(self) -> None:
        manifest = read_json(Path("tests/fixtures/positive/manifest.v1.example.json"))
        manifest["runtime"]["strict_hardware"] = False
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            manifest_path = tdir / "manifest.nonstrict.json"
            write_json(manifest_path, manifest)

            observed = copy.deepcopy(manifest["hardware_profile"])
            observed["gpu"]["driver_version"] = "551.00.00"
            runtime_hw = tdir / "runtime.hardware.json"
            write_json(runtime_hw, observed)

            resolved = tdir / "resolved.lock.json"
            built = tdir / "built.lock.json"
            run_base = tdir / "run-base"
            run_candidate = tdir / "run-candidate"
            report = tdir / "verify_report.json"
            summary = tdir / "verify_summary.txt"

            run_cmd([sys.executable, "modules/inference/resolver/main.py", "--manifest", str(manifest_path), "--lockfile-out", str(resolved)])
            run_cmd([sys.executable, "modules/build/builder/main.py", "--lockfile", str(resolved), "--lockfile-out", str(built)])

            run_cmd(
                [
                    sys.executable,
                    "modules/inference/runner/main.py", "--mode", "mock",
                    "--manifest",
                    str(manifest_path),
                    "--lockfile",
                    str(built),
                    "--out-dir",
                    str(run_base),
                ]
            )
            run_cmd(
                [
                    sys.executable,
                    "modules/inference/runner/main.py", "--mode", "mock",
                    "--manifest",
                    str(manifest_path),
                    "--lockfile",
                    str(built),
                    "--out-dir",
                    str(run_candidate),
                    "--runtime-hardware",
                    str(runtime_hw),
                ]
            )

            candidate_bundle = read_json(run_candidate / "run_bundle.v1.json")
            self.assertEqual(candidate_bundle["hardware_conformance"]["status"], "non_conformant")
            self.assertFalse(candidate_bundle["hardware_conformance"]["strict_hardware"])
            self.assertGreater(len(candidate_bundle["hardware_conformance"]["diffs"]), 0)
            self.assertEqual(
                candidate_bundle["environment_info"]["hardware_fingerprint"],
                candidate_bundle["hardware_conformance"]["actual_fingerprint"],
            )

            run_cmd(
                [
                    sys.executable,
                    "modules/attestation/verifier/main.py",
                    "--baseline",
                    str(run_base / "run_bundle.v1.json"),
                    "--candidate",
                    str(run_candidate / "run_bundle.v1.json"),
                    "--report-out",
                    str(report),
                    "--summary-out",
                    str(summary),
                ]
            )
            verify = read_json(report)
            self.assertEqual(verify["status"], "non_conformant_hardware")


if __name__ == "__main__":
    unittest.main()
