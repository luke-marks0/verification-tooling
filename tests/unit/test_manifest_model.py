"""Unit tests for the Pydantic manifest model."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Insert repo root so modules.inference.manifest is importable.
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydantic import ValidationError as PydanticValidationError

from modules.inference.manifest.model import (
    ArtifactInput,
    ArtifactType,
    Comparator,
    ComparisonConfig,
    ComparisonMode,
    Manifest,
    RequestItem,
)


def _load_json(relpath: str) -> dict:
    return json.loads((REPO_ROOT / relpath).read_text(encoding="utf-8"))


REAL_MANIFEST = _load_json("modules/inference/manifests/qwen3-1.7b.manifest.json")
POSITIVE_FIXTURE = _load_json("tests/fixtures/positive/manifest.v1.example.json")


class TestManifestParsesRealManifest(unittest.TestCase):
    """The Pydantic model must parse the real qwen3-1.7b manifest."""

    def test_parses_real_manifest(self) -> None:
        m = Manifest.model_validate(REAL_MANIFEST)
        self.assertEqual(m.model.source, "hf://Qwen/Qwen3-1.7B")
        self.assertEqual(m.model.weights_revision, "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
        self.assertEqual(m.model.trust_remote_code, False)
        self.assertEqual(m.runtime.strict_hardware, False)
        self.assertEqual(m.runtime.serving_engine.max_model_len, 8192)
        self.assertEqual(m.runtime.deterministic_knobs.seed, 42)
        self.assertEqual(m.runtime.batch_invariance.enabled, True)
        self.assertEqual(m.runtime.batch_invariance.enforce_eager, True)
        self.assertEqual(m.hardware_profile.gpu.model, "NVIDIA GH200 480GB")
        self.assertEqual(m.hardware_profile.gpu.count, 1)
        self.assertEqual(m.hardware_profile.gpu.driver_version, "570.148.08")
        self.assertEqual(m.hardware_profile.gpu.cuda_driver_version, "12.8")
        self.assertGreaterEqual(len(m.requests), 1)
        self.assertGreaterEqual(len(m.artifact_inputs), 4)
        self.assertEqual(m.comparison.tokens.mode, ComparisonMode.exact)
        self.assertEqual(m.comparison.logits.mode, ComparisonMode.absrel)
        self.assertIsNone(m.runtime.closure_hash)

    def test_parses_positive_fixture(self) -> None:
        m = Manifest.model_validate(POSITIVE_FIXTURE)
        self.assertEqual(m.manifest_version, "v1")
        self.assertEqual(m.run_id, "run-0001")
        self.assertEqual(m.model.source, "hf://org/model")
        self.assertEqual(m.runtime.strict_hardware, True)

    def test_roundtrip_model_dump(self) -> None:
        """model_validate(d).model_dump() should reproduce the input (modulo defaults)."""
        m = Manifest.model_validate(REAL_MANIFEST)
        dumped = m.model_dump(exclude_none=True)
        # Re-parse the dump to ensure it validates
        m2 = Manifest.model_validate(dumped)
        self.assertEqual(m.model.source, m2.model.source)
        self.assertEqual(m.run_id, m2.run_id)


class TestManifestRejectsInvalid(unittest.TestCase):
    """The Pydantic model must reject manifests that violate the schema."""

    def _mutate(self, **overrides: object) -> dict:
        d = copy.deepcopy(REAL_MANIFEST)
        for dotpath, value in overrides.items():
            parts = dotpath.split(".")
            target = d
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        return d

    def test_rejects_missing_run_id(self) -> None:
        d = copy.deepcopy(REAL_MANIFEST)
        del d["run_id"]
        with self.assertRaises(PydanticValidationError):
            Manifest.model_validate(d)

    def test_rejects_bad_manifest_version(self) -> None:
        with self.assertRaises(PydanticValidationError):
            Manifest.model_validate(self._mutate(**{"manifest_version": "v2"}))

    def test_rejects_extra_top_level_field(self) -> None:
        d = copy.deepcopy(REAL_MANIFEST)
        d["network"] = {"mtu": 1500}
        with self.assertRaises(PydanticValidationError):
            Manifest.model_validate(d)

    def test_rejects_extra_gpu_field(self) -> None:
        d = copy.deepcopy(REAL_MANIFEST)
        d["hardware_profile"]["gpu"]["vendor"] = "nvidia"
        with self.assertRaises(PydanticValidationError):
            Manifest.model_validate(d)

    def test_rejects_bad_weights_revision(self) -> None:
        with self.assertRaises(PydanticValidationError):
            Manifest.model_validate(self._mutate(**{"model.weights_revision": "not-a-sha"}))

    def test_rejects_negative_seed(self) -> None:
        with self.assertRaises(PydanticValidationError):
            Manifest.model_validate(self._mutate(**{"runtime.deterministic_knobs.seed": -1}))

    def test_rejects_empty_requests(self) -> None:
        d = copy.deepcopy(REAL_MANIFEST)
        d["requests"] = []
        with self.assertRaises(PydanticValidationError):
            Manifest.model_validate(d)

    def test_accepts_empty_artifacts(self) -> None:
        d = copy.deepcopy(REAL_MANIFEST)
        d["artifact_inputs"] = []
        m = Manifest.model_validate(d)
        self.assertEqual(len(m.artifact_inputs), 0)

    def test_rejects_bad_created_at(self) -> None:
        with self.assertRaises(PydanticValidationError):
            Manifest.model_validate(self._mutate(**{"created_at": "not-a-date"}))


class TestSubModels(unittest.TestCase):
    """Test individual sub-models in isolation."""

    def test_comparator_exact(self) -> None:
        c = Comparator(mode=ComparisonMode.exact)
        self.assertEqual(c.mode, ComparisonMode.exact)
        self.assertIsNone(c.atol)

    def test_comparator_absrel(self) -> None:
        c = Comparator(mode=ComparisonMode.absrel, atol=1e-6, rtol=1e-4)
        self.assertEqual(c.atol, 1e-6)
        self.assertEqual(c.rtol, 1e-4)

    def test_comparator_hash_valid(self) -> None:
        c = Comparator(mode=ComparisonMode.hash, algorithm="sha256")
        self.assertEqual(c.algorithm, "sha256")

    def test_comparator_ulp_valid(self) -> None:
        c = Comparator(mode=ComparisonMode.ulp, ulp=2)
        self.assertEqual(c.ulp, 2)

    def test_comparator_hash_requires_algorithm(self) -> None:
        with self.assertRaises(PydanticValidationError):
            Comparator(mode=ComparisonMode.hash)

    def test_comparator_ulp_requires_ulp(self) -> None:
        with self.assertRaises(PydanticValidationError):
            Comparator(mode=ComparisonMode.ulp)

    def test_comparator_absrel_requires_atol_rtol(self) -> None:
        with self.assertRaises(PydanticValidationError):
            Comparator(mode=ComparisonMode.absrel, atol=1e-6)

    def test_artifact_input_minimal(self) -> None:
        a = ArtifactInput(
            artifact_id="test-art",
            artifact_type=ArtifactType.model_weights,
            source_kind="hf",
            source_uri="hf://org/model/weights.safetensors",
            immutable_ref="a" * 40,
        )
        self.assertEqual(a.artifact_type, ArtifactType.model_weights)
        self.assertIsNone(a.expected_digest)
        self.assertIsNone(a.size_bytes)

    def test_request_item(self) -> None:
        r = RequestItem(id="req-1", prompt="Hello", max_new_tokens=8, temperature=0)
        self.assertEqual(r.id, "req-1")
        self.assertEqual(r.temperature, 0)


if __name__ == "__main__":
    unittest.main()
