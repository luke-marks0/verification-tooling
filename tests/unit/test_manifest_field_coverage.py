"""Schema-driven manifest field coverage test.

Walks the manifest JSON Schema recursively, collects every field path,
then asserts that each path is exercised by at least one fixture or
test-constructed manifest that successfully parses through the Pydantic model.

If a new field is added to the schema but no fixture or coverage manifest
exercises it, this test fails — preventing silent coverage drift.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.inference.manifest.model import Manifest

SCHEMA_PATH = REPO_ROOT / "modules" / "core" / "schemas" / "manifest.v1.schema.json"
POSITIVE_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "positive" / "manifest.v1.example.json"


# ---------------------------------------------------------------------------
# Schema walking: extract every leaf + container path from the JSON Schema
# ---------------------------------------------------------------------------

def _collect_schema_paths(
    schema: dict,
    prefix: str = "",
    defs: dict | None = None,
) -> set[str]:
    """Return the set of dot-separated field paths defined by *schema*."""
    if defs is None:
        defs = schema.get("$defs", {})

    paths: set[str] = set()

    # Resolve $ref
    if "$ref" in schema:
        ref = schema["$ref"]  # e.g. "#/$defs/comparator"
        ref_name = ref.rsplit("/", 1)[-1]
        return _collect_schema_paths(defs[ref_name], prefix, defs)

    schema_type = schema.get("type")

    # Unwrap type arrays like ["string", "null"] → treat as the non-null type
    if isinstance(schema_type, list):
        schema_type = next((t for t in schema_type if t != "null"), schema_type[0])

    if schema_type == "object" and "properties" in schema:
        if prefix:
            paths.add(prefix)
        for prop_name, prop_schema in schema["properties"].items():
            child = f"{prefix}.{prop_name}" if prefix else prop_name
            paths |= _collect_schema_paths(prop_schema, child, defs)
    elif schema_type == "array" and "items" in schema:
        if prefix:
            paths.add(prefix)
        paths |= _collect_schema_paths(schema["items"], f"{prefix}[]", defs)
    else:
        # Leaf field
        if prefix:
            paths.add(prefix)

    return paths


# ---------------------------------------------------------------------------
# Instance walking: extract every field path present in a manifest dict
# ---------------------------------------------------------------------------

def _collect_instance_paths(obj: object, prefix: str = "") -> set[str]:
    """Return the set of dot-separated field paths present in *obj*."""
    paths: set[str] = set()
    if isinstance(obj, dict):
        if prefix:
            paths.add(prefix)
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else key
            paths |= _collect_instance_paths(value, child)
    elif isinstance(obj, list):
        if prefix:
            paths.add(prefix)
        for item in obj:
            paths |= _collect_instance_paths(item, f"{prefix}[]")
    else:
        if prefix:
            paths.add(prefix)
    return paths


# ---------------------------------------------------------------------------
# Coverage manifests — one per comparator mode and source_kind gap
# ---------------------------------------------------------------------------

def _base_manifest() -> dict:
    """Minimal valid manifest with all required fields."""
    return {
        "manifest_version": "v1",
        "run_id": "coverage-test-001",
        "created_at": "2026-03-27T00:00:00Z",
        "model": {
            "source": "hf://org/model",
            "weights_revision": "a" * 40,
            "tokenizer_revision": "b" * 40,
            "trust_remote_code": True,
        },
        "runtime": {
            "strict_hardware": False,
            "batch_invariance": {"enabled": True, "enforce_eager": True},
            "deterministic_knobs": {
                "seed": 42,
                "torch_deterministic": True,
                "cuda_launch_blocking": True,
                "cublas_workspace_config": ":4096:8",
                "pythonhashseed": "0",
            },
            "serving_engine": {
                "max_model_len": 4096,
                "max_num_seqs": 128,
                "gpu_memory_utilization": 0.9,
                "dtype": "bfloat16",
                "attention_backend": "FLASH_ATTN",
                "quantization": "awq",
                "load_format": "safetensors",
                "kv_cache_dtype": "fp8",
                "max_num_batched_tokens": 4096,
                "block_size": 16,
                "enable_prefix_caching": True,
                "enable_chunked_prefill": False,
                "scheduling_policy": "fcfs",
                "disable_sliding_window": False,
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 1,
                "disable_custom_all_reduce": False,
            },
            "closure_hash": "sha256:" + "ab" * 32,
        },
        "hardware_profile": {
            "gpu": {
                "model": "H100-SXM-80GB",
                "count": 2,
                "driver_version": "550.54.15",
                "cuda_driver_version": "12.4",
            }
        },
        "requests": [
            {"id": "req-1", "prompt": "Hello world", "max_new_tokens": 64, "temperature": 0.7},
        ],
        "comparison": {
            "tokens": {"mode": "exact"},
            "logits": {"mode": "absrel", "atol": 1e-6, "rtol": 1e-4},
        },
        "artifact_inputs": [],
    }


def _build_coverage_manifests() -> list[dict]:
    """Return manifests that collectively exercise every schema path."""
    manifests = []

    # 1. Base manifest with all optional serving_engine + deterministic_knobs fields
    manifests.append(_base_manifest())

    # 2. Manifest exercising ulp (tokens) and hash (logits) comparator modes
    m_ulp = _base_manifest()
    m_ulp["run_id"] = "coverage-ulp-hash-001"
    m_ulp["comparison"] = {
        "tokens": {"mode": "ulp", "ulp": 2},
        "logits": {"mode": "hash", "algorithm": "sha256"},
        "network_egress": {"mode": "exact"},
    }
    manifests.append(m_ulp)

    # 3. Manifest exercising hash (tokens) and ulp (logits) — covers cross fields
    m_cross = _base_manifest()
    m_cross["run_id"] = "coverage-cross-comp-001"
    m_cross["comparison"] = {
        "tokens": {"mode": "hash", "algorithm": "sha256"},
        "logits": {"mode": "ulp", "ulp": 4},
        "network_egress": {"mode": "ulp", "ulp": 1},
    }
    manifests.append(m_cross)

    # 4. Manifest exercising absrel on tokens and exact on logits
    m_absrel_tok = _base_manifest()
    m_absrel_tok["run_id"] = "coverage-absrel-tok-001"
    m_absrel_tok["comparison"] = {
        "tokens": {"mode": "absrel", "atol": 1e-5, "rtol": 1e-3},
        "logits": {"mode": "exact"},
        "network_egress": {"mode": "absrel", "atol": 1e-6, "rtol": 1e-4},
    }
    manifests.append(m_absrel_tok)

    # 3. Manifest exercising all artifact_type values, all source_kind values,
    #    and all optional artifact fields including every role value
    all_artifact_types = [
        "model_weights", "model_config", "tokenizer", "generation_config",
        "chat_template", "prompt_formatter", "serving_stack", "container_image",
        "cuda_lib", "kernel_library", "network_stack_binary", "pmd_driver",
        "runtime_knob_set", "request_set", "batching_policy", "nic_link_config",
        "collective_stack", "compiled_extension", "remote_code",
    ]
    all_source_kinds = ["hf", "oci", "s3", "http", "git", "nix", "inline"]
    all_roles = [
        "weights_shard", "config", "tokenizer",
        "generation_config", "chat_template", "prompt_formatter",
    ]

    artifacts = []
    for i, art_type in enumerate(all_artifact_types):
        source_kind = all_source_kinds[i % len(all_source_kinds)]
        uri_map = {
            "hf": "hf://org/model/file.bin",
            "oci": "oci://registry.example/img@sha256:" + "aa" * 32,
            "s3": "s3://bucket/key",
            "http": "https://example.com/file.bin",
            "git": "git://github.com/org/repo",
            "nix": "nix://nixpkgs#hello",
            "inline": "inline://data",
        }
        art: dict = {
            "artifact_id": f"art-{art_type}",
            "artifact_type": art_type,
            "source_kind": source_kind,
            "source_uri": uri_map[source_kind],
            "immutable_ref": "a" * 40,
            "name": f"coverage-{art_type}",
            "expected_digest": "sha256:" + "cc" * 32,
            "size_bytes": 1000 + i,
            "path": f"/data/{art_type}/file.bin",
        }
        # Assign roles to the first N artifacts that support them
        if i < len(all_roles):
            art["role"] = all_roles[i]
        artifacts.append(art)

    m_arts = _base_manifest()
    m_arts["run_id"] = "coverage-artifacts-001"
    m_arts["artifact_inputs"] = artifacts
    manifests.append(m_arts)

    # Manifests exercising remaining serving_engine enum values
    enum_combos = [
        {
            "run_id": "coverage-enum-001",
            "dtype": "float16",
            "attention_backend": "TRITON_MLA",
            "load_format": "pt",
            "kv_cache_dtype": "int8",
            "quantization": "gptq",
            "scheduling_policy": "priority",
        },
        {
            "run_id": "coverage-enum-002",
            "dtype": "float32",
            "attention_backend": "FLASH_ATTN_MLA",
            "load_format": "gguf",
            "kv_cache_dtype": "auto",
            "quantization": "fp8",
        },
        {
            "run_id": "coverage-enum-003",
            "dtype": "bfloat16",
            "attention_backend": "TRITON_ATTN",
            "load_format": "auto",
            "quantization": "bitsandbytes",
        },
    ]
    for combo in enum_combos:
        m_enum = _base_manifest()
        m_enum["run_id"] = combo.pop("run_id")
        m_enum["runtime"]["serving_engine"].update(combo)
        manifests.append(m_enum)

    # Manifest exercising the optional audit block
    m_audit = _base_manifest()
    m_audit["run_id"] = "coverage-audit-001"
    m_audit["audit"] = {
        "token_commitment": {
            "enabled": True,
            "algorithm": "hmac-sha256",
            "key_source": "inline-shared",
        }
    }
    manifests.append(m_audit)

    # Manifests exercising distributed_executor_backend values
    for backend, run_id in (("ray", "coverage-dist-ray-001"), ("mp", "coverage-dist-mp-001")):
        m_dist = _base_manifest()
        m_dist["run_id"] = run_id
        m_dist["runtime"]["serving_engine"]["distributed_executor_backend"] = backend
        manifests.append(m_dist)

    return manifests


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestManifestFieldCoverage(unittest.TestCase):
    """Every field path in the manifest JSON schema must be exercised."""

    @classmethod
    def setUpClass(cls) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.schema_paths = _collect_schema_paths(schema)

        # Collect covered paths from all sources
        cls.covered_paths: set[str] = set()

        # Source 1: positive fixture
        fixture = json.loads(POSITIVE_FIXTURE.read_text(encoding="utf-8"))
        Manifest.model_validate(fixture)  # must parse
        cls.covered_paths |= _collect_instance_paths(fixture)

        # Source 2: real manifest (if present)
        real_path = REPO_ROOT / "modules" / "inference" / "manifests" / "qwen3-1.7b.manifest.json"
        if real_path.exists():
            real = json.loads(real_path.read_text(encoding="utf-8"))
            Manifest.model_validate(real)
            cls.covered_paths |= _collect_instance_paths(real)

        # Source 3: coverage manifests built to fill gaps
        for m_dict in _build_coverage_manifests():
            Manifest.model_validate(m_dict)  # must parse
            cls.covered_paths |= _collect_instance_paths(m_dict)

    def test_all_schema_paths_covered(self) -> None:
        """Every field path in the schema must appear in at least one manifest."""
        uncovered = self.schema_paths - self.covered_paths
        if uncovered:
            sorted_uncovered = sorted(uncovered)
            self.fail(
                f"{len(sorted_uncovered)} schema field path(s) not exercised "
                f"by any fixture or coverage manifest:\n"
                + "\n".join(f"  - {p}" for p in sorted_uncovered)
            )

    def test_coverage_manifests_parse(self) -> None:
        """All coverage manifests must parse through the Pydantic model."""
        for i, m_dict in enumerate(_build_coverage_manifests()):
            with self.subTest(manifest=i):
                m = Manifest.model_validate(m_dict)
                self.assertEqual(m.manifest_version, "v1")

    def test_every_comparator_mode_covered(self) -> None:
        """Each comparison mode (exact, ulp, absrel, hash) must appear."""
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        comparator_def = schema["$defs"]["comparator"]
        modes = set(comparator_def["properties"]["mode"]["enum"])

        covered_modes: set[str] = set()
        for m_dict in _build_coverage_manifests():
            for field in ("tokens", "logits"):
                covered_modes.add(m_dict["comparison"][field]["mode"])
        # Also check fixture
        fixture = json.loads(POSITIVE_FIXTURE.read_text(encoding="utf-8"))
        for field in ("tokens", "logits"):
            covered_modes.add(fixture["comparison"][field]["mode"])

        missing = modes - covered_modes
        if missing:
            self.fail(f"Comparator modes not exercised: {sorted(missing)}")

    def test_every_artifact_type_covered(self) -> None:
        """Each artifact_type enum value must appear in at least one manifest."""
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        all_types = set(schema["$defs"]["artifact_type"]["enum"])

        covered_types: set[str] = set()
        for m_dict in _build_coverage_manifests():
            for art in m_dict.get("artifact_inputs", []):
                covered_types.add(art["artifact_type"])
        fixture = json.loads(POSITIVE_FIXTURE.read_text(encoding="utf-8"))
        for art in fixture.get("artifact_inputs", []):
            covered_types.add(art["artifact_type"])

        missing = all_types - covered_types
        if missing:
            self.fail(f"Artifact types not exercised: {sorted(missing)}")

    def test_every_source_kind_covered(self) -> None:
        """Each source_kind enum value must appear in at least one manifest."""
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        items_props = schema["properties"]["artifact_inputs"]["items"]["properties"]
        all_kinds = set(items_props["source_kind"]["enum"])

        covered_kinds: set[str] = set()
        for m_dict in _build_coverage_manifests():
            for art in m_dict.get("artifact_inputs", []):
                covered_kinds.add(art["source_kind"])
        fixture = json.loads(POSITIVE_FIXTURE.read_text(encoding="utf-8"))
        for art in fixture.get("artifact_inputs", []):
            covered_kinds.add(art["source_kind"])

        missing = all_kinds - covered_kinds
        if missing:
            self.fail(f"Source kinds not exercised: {sorted(missing)}")

    def test_every_enum_value_covered(self) -> None:
        """Every enum value across the entire schema must appear in at least one manifest."""
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        # Collect all enum paths and their allowed values from the schema
        enum_values: dict[str, set[str]] = {}
        self._find_enums(schema, "", schema.get("$defs", {}), enum_values)

        # Collect all values that appear in coverage manifests + fixtures
        all_manifests = list(_build_coverage_manifests())
        all_manifests.append(json.loads(POSITIVE_FIXTURE.read_text(encoding="utf-8")))
        real_path = REPO_ROOT / "modules" / "inference" / "manifests" / "qwen3-1.7b.manifest.json"
        if real_path.exists():
            all_manifests.append(json.loads(real_path.read_text(encoding="utf-8")))

        covered_values: dict[str, set[str]] = {path: set() for path in enum_values}
        for m_dict in all_manifests:
            self._collect_enum_values(m_dict, "", covered_values)

        # Report uncovered
        missing_lines = []
        for path in sorted(enum_values):
            uncovered = enum_values[path] - covered_values.get(path, set())
            # Filter out None/null — not representable as string in simple collection
            uncovered = {v for v in uncovered if v is not None}
            if uncovered:
                missing_lines.append(f"  {path}: {sorted(uncovered)}")

        if missing_lines:
            self.fail(
                "Enum values not exercised:\n" + "\n".join(missing_lines)
            )

    # -- helpers for enum coverage --

    def _find_enums(
        self, schema: dict, prefix: str, defs: dict,
        out: dict[str, set[str]],
    ) -> None:
        if "$ref" in schema:
            ref_name = schema["$ref"].rsplit("/", 1)[-1]
            self._find_enums(defs[ref_name], prefix, defs, out)
            return
        if "enum" in schema:
            out[prefix] = set(str(v) if v is not None else None for v in schema["enum"])
        if "properties" in schema:
            for prop, prop_schema in schema["properties"].items():
                child = f"{prefix}.{prop}" if prefix else prop
                self._find_enums(prop_schema, child, defs, out)
        if "items" in schema:
            self._find_enums(schema["items"], f"{prefix}[]", defs, out)

    def _collect_enum_values(
        self, obj: object, prefix: str,
        out: dict[str, set[str]],
    ) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                child = f"{prefix}.{key}" if prefix else key
                if child in out and isinstance(value, (str, int, float)):
                    out[child].add(str(value))
                self._collect_enum_values(value, child, out)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_enum_values(item, f"{prefix}[]", out)


if __name__ == "__main__":
    unittest.main()
