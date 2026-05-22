"""Phase 2 facade smoke tests: attestation, utils, build, memory + verified_inference.

All CPU / no-GPU. The nix and PoSE-hardware paths are exercised only as far as
importability + the parts that run anywhere.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANIFEST = "tests/fixtures/positive/manifest.v1.example.json"


class TestAttestationFacade(unittest.TestCase):
    def test_honest_matmul_round_trip_passes(self) -> None:
        from modules.attestation import Challenge, ComparisonMode, MatmulSpec, attest_matmuls

        challenge = Challenge(
            challenge_id="facade-test-001",
            matmuls=(
                MatmulSpec(
                    id="m0", M=4, K=6, N=5,
                    dtype_a="int8", dtype_b="int8", dtype_acc="int32", dtype_c="int32",
                    seed_a=10, seed_b=11, comparison=ComparisonMode.BITWISE,
                ),
            ),
        )
        report = attest_matmuls(challenge)
        self.assertTrue(report.overall_passed)

    def test_token_commitment_is_deterministic(self) -> None:
        from modules.attestation import commit_token

        self.assertEqual(commit_token(42), commit_token(42))
        self.assertNotEqual(commit_token(42), commit_token(43))


class TestUtilsFacade(unittest.TestCase):
    def test_canonical_json_and_digest(self) -> None:
        from modules.utils import canonical_json_bytes, sha256_prefixed

        a = canonical_json_bytes({"b": 1, "a": 2})
        b = canonical_json_bytes({"a": 2, "b": 1})
        self.assertEqual(a, b)  # key order independent
        self.assertTrue(sha256_prefixed(a).startswith("sha256:"))


class TestBuildFacade(unittest.TestCase):
    def test_build_runtime_enriches_lockfile(self) -> None:
        # build_runtime is the pure-Python layer (no nix); exercise via Pipeline.
        from modules import Pipeline

        pipe = Pipeline.from_manifest(MANIFEST).resolve().build()
        assert pipe.lockfile is not None
        self.assertIn("build", pipe.lockfile)
        self.assertTrue(pipe.lockfile["runtime_closure_digest"].startswith("sha256:"))


class TestMemoryFacade(unittest.TestCase):
    def test_pose_path_resolves_and_imports(self) -> None:
        from modules.memory import POSE_SRC, load_pose

        self.assertTrue(POSE_SRC.is_dir())
        pose = load_pose()  # top-level only; no GPU/crypto backends
        self.assertTrue(hasattr(pose, "hello"))


class TestVerifiedInferenceWorkflow(unittest.TestCase):
    def test_verified_inference_synthetic(self) -> None:
        from workflows.verified_inference import verified_inference

        result = verified_inference(MANIFEST, mode="synthetic")
        self.assertEqual(result["run_status"], "conformant")
        self.assertTrue(result["attestation_passed"])


if __name__ == "__main__":
    unittest.main()
