from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import read_json, run_cmd


class TestRunnerContextProvenance(unittest.TestCase):
    def test_runner_records_kubernetes_context_and_rerun_metadata(self) -> None:
        manifest = "tests/fixtures/positive/manifest.v1.example.json"
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            lock_resolved = tdir / "resolved.lock.json"
            lock_built = tdir / "built.lock.json"
            run_dir = tdir / "run"

            run_cmd(["python3", "modules/inference/resolver/main.py", "--manifest", manifest, "--lockfile-out", str(lock_resolved)])
            run_cmd(["python3", "modules/build/builder/main.py", "--lockfile", str(lock_resolved), "--lockfile-out", str(lock_built)])
            run_cmd(
                ["python3", "modules/inference/runner/main.py", "--manifest", manifest, "--lockfile", str(lock_built), "--out-dir", str(run_dir)],
                env={
                    "RUNNER_POD_MANIFEST_PATH": "/run-inputs/manifest.json",
                    "RUNNER_POD_LOCKFILE_PATH": "/run-inputs/lockfile.json",
                    "RUNNER_POD_RUNTIME_CLOSURE_PATH": "/run-inputs/runtime_closure_digest.txt",
                    "RUNNER_POD_NAME": "runner-0",
                    "RUNNER_NODE_NAME": "node-h100-0",
                    "RUNNER_NAMESPACE": "deterministic-serving",
                    "RUNNER_GPU_MODEL": "H100-SXM-80GB",
                    "RUNNER_NIC_MODEL": "ConnectX-7",
                },
            )

            bundle = read_json(run_dir / "run_bundle.v1.json")
            built_lockfile = read_json(lock_built)
            self.assertEqual(bundle["hardware_probe"]["source"], "env_probe")
            self.assertEqual(bundle["manifest_copy"]["path"], "manifest.json")
            self.assertEqual(bundle["lockfile_copy"]["path"], "lockfile.json")
            self.assertEqual(bundle["manifest_copy"]["digest"], bundle["rerun_metadata"]["manifest_digest"])
            self.assertEqual(bundle["runtime_closure_digest"], built_lockfile["runtime_closure_digest"])
            self.assertGreaterEqual(len(bundle["resolved_artifact_digests"]), 1)
            self.assertGreaterEqual(len(bundle["attestations"]), 1)
            self.assertIn("tokens", bundle["observables"])
            self.assertIn("logits", bundle["observables"])
            self.assertEqual(bundle["execution_context"]["pod"]["name"], "runner-0")
            self.assertEqual(bundle["execution_context"]["pod"]["node_name"], "node-h100-0")
            self.assertEqual(bundle["execution_context"]["pod"]["namespace"], "deterministic-serving")
            self.assertEqual(bundle["execution_context"]["input_mounts"]["manifest_path"], "/run-inputs/manifest.json")
            self.assertEqual(bundle["execution_context"]["input_mounts"]["lockfile_path"], "/run-inputs/lockfile.json")
            self.assertEqual(
                bundle["execution_context"]["input_mounts"]["runtime_closure_path"],
                "/run-inputs/runtime_closure_digest.txt",
            )
            self.assertEqual(bundle["rerun_metadata"]["manifest_digest"], bundle["manifest_copy"]["digest"])
            self.assertEqual(
                bundle["rerun_metadata"]["lockfile_digest"],
                built_lockfile["canonicalization"]["lockfile_digest"],
            )
            self.assertEqual(bundle["rerun_metadata"]["runtime_closure_digest"], bundle["runtime_closure_digest"])
            self.assertGreaterEqual(bundle["rerun_metadata"]["artifact_count"], 1)
            self.assertGreaterEqual(len(bundle["rerun_metadata"]["attestation_digests"]), 1)


if __name__ == "__main__":
    unittest.main()
