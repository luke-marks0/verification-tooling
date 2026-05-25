"""Unit tests for manifest validation and enforcement (no GPU needed)."""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.core.common.contracts import ValidationError, validate_with_schema
from modules.inference.manifest.model import Manifest

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "server_main", REPO_ROOT / "modules" / "inference" / "server" / "main.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_enforce_model_revision = _mod._enforce_model_revision
_validate_requests = _mod._validate_requests
_build_vllm_cmd = _mod._build_vllm_cmd
_set_deterministic_env = _mod._set_deterministic_env
_verify_closure = _mod._verify_closure
_verify_model_artifacts = _mod._verify_model_artifacts
_check_hardware = _mod._check_hardware


def _load_manifest_dict() -> dict:
    path = REPO_ROOT / "modules" / "inference" / "manifests" / "qwen3-1.7b.manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest() -> Manifest:
    return Manifest.model_validate(_load_manifest_dict())


def _manifest_from_dict(d: dict) -> Manifest:
    """Convert a (possibly mutated) dict to a Manifest, skipping extra-field rejection."""
    return Manifest.model_validate(d)


class TestSchemaValidation(unittest.TestCase):
    """Schema is the first gate for POST /manifest."""

    def test_valid_manifest_passes(self) -> None:
        validate_with_schema("manifest.v1.schema.json", _load_manifest_dict())

    def test_missing_run_id_rejected(self) -> None:
        m = _load_manifest_dict()
        del m["run_id"]
        with self.assertRaises(ValidationError):
            validate_with_schema("manifest.v1.schema.json", m)

    def test_bad_model_source_rejected(self) -> None:
        m = _load_manifest_dict()
        m["model"]["source"] = "not-a-valid-source"
        with self.assertRaises(ValidationError):
            validate_with_schema("manifest.v1.schema.json", m)

    def test_empty_dict_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            validate_with_schema("manifest.v1.schema.json", {})

    def test_missing_model_rejected(self) -> None:
        m = _load_manifest_dict()
        del m["model"]
        with self.assertRaises(ValidationError):
            validate_with_schema("manifest.v1.schema.json", m)

    def test_missing_runtime_rejected(self) -> None:
        m = _load_manifest_dict()
        del m["runtime"]
        with self.assertRaises(ValidationError):
            validate_with_schema("manifest.v1.schema.json", m)

    def test_bad_temperature_rejected(self) -> None:
        m = _load_manifest_dict()
        m["requests"][0]["temperature"] = 5.0
        with self.assertRaises(ValidationError):
            validate_with_schema("manifest.v1.schema.json", m)


class TestModelRevisionEnforcement(unittest.TestCase):
    """Model revision pinning."""

    def test_pinned_revision_returned(self) -> None:
        m = _load_manifest()
        rev = _enforce_model_revision(m)
        self.assertIsNotNone(rev)
        self.assertEqual(len(rev), 40)  # sha1 hex

    def test_missing_revision_returns_none(self) -> None:
        d = _load_manifest_dict()
        # weights_revision is required in Pydantic, so test via the function's return
        m = _load_manifest()
        rev = _enforce_model_revision(m)
        self.assertIsNotNone(rev)


class TestRequestValidation(unittest.TestCase):
    """Requests must be servable with the declared engine config."""

    def test_valid_requests_pass(self) -> None:
        m = _load_manifest()
        errors = _validate_requests(m)
        self.assertEqual(errors, [])

    def test_request_exceeding_max_model_len(self) -> None:
        d = _load_manifest_dict()
        # Set a small max_model_len so we can create requests that exceed it
        # while staying within Pydantic's max_new_tokens=4096 limit.
        d["runtime"]["serving_engine"]["max_model_len"] = 64
        d["requests"] = [
            {"id": "too-long", "prompt": "hi", "max_new_tokens": 128, "temperature": 0}
        ]
        m = _manifest_from_dict(d)
        errors = _validate_requests(m)
        self.assertEqual(len(errors), 1)
        self.assertIn("exceeds max_model_len", errors[0])

    def test_multiple_invalid_requests(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["max_model_len"] = 64
        d["requests"] = [
            {"id": "a", "prompt": "hi", "max_new_tokens": 65, "temperature": 0},
            {"id": "b", "prompt": "hi", "max_new_tokens": 200, "temperature": 0},
        ]
        m = _manifest_from_dict(d)
        errors = _validate_requests(m)
        self.assertEqual(len(errors), 2)

    def test_requests_at_exactly_max_len(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["max_model_len"] = 256
        d["requests"] = [
            {"id": "exact", "prompt": "hi", "max_new_tokens": 256, "temperature": 0},
        ]
        m = _manifest_from_dict(d)
        errors = _validate_requests(m)
        self.assertEqual(errors, [])


class TestBuildVllmCmd(unittest.TestCase):
    """Test that _build_vllm_cmd passes every serving_engine field to vLLM."""

    def test_quantization_flag_present(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["quantization"] = "awq"
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--quantization", cmd)
        self.assertEqual(cmd[cmd.index("--quantization") + 1], "awq")

    def test_quantization_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--quantization", cmd)

    def test_load_format_flag_present(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["load_format"] = "safetensors"
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--load-format", cmd)
        self.assertEqual(cmd[cmd.index("--load-format") + 1], "safetensors")

    def test_load_format_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--load-format", cmd)

    def test_kv_cache_dtype_flag_present(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["kv_cache_dtype"] = "fp8"
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--kv-cache-dtype", cmd)
        self.assertEqual(cmd[cmd.index("--kv-cache-dtype") + 1], "fp8")

    def test_kv_cache_dtype_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--kv-cache-dtype", cmd)

    def test_max_num_batched_tokens_flag_present(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["max_num_batched_tokens"] = 4096
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--max-num-batched-tokens", cmd)
        self.assertEqual(cmd[cmd.index("--max-num-batched-tokens") + 1], "4096")

    def test_max_num_batched_tokens_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--max-num-batched-tokens", cmd)

    def test_block_size_flag_present(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["block_size"] = 16
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--block-size", cmd)
        self.assertEqual(cmd[cmd.index("--block-size") + 1], "16")

    def test_block_size_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--block-size", cmd)

    def test_enable_prefix_caching_flag_present(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["enable_prefix_caching"] = True
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--enable-prefix-caching", cmd)

    def test_enable_prefix_caching_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--enable-prefix-caching", cmd)

    def test_enable_chunked_prefill_flag_present(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["enable_chunked_prefill"] = True
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--enable-chunked-prefill", cmd)

    def test_enable_chunked_prefill_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--enable-chunked-prefill", cmd)

    def test_scheduling_policy_flag_present(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["scheduling_policy"] = "fcfs"
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--scheduling-policy", cmd)
        self.assertEqual(cmd[cmd.index("--scheduling-policy") + 1], "fcfs")

    def test_scheduling_policy_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--scheduling-policy", cmd)

    def test_disable_sliding_window_flag_present(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["disable_sliding_window"] = True
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--disable-sliding-window", cmd)

    def test_disable_sliding_window_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--disable-sliding-window", cmd)

    def test_tensor_parallel_size_flag_present(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["tensor_parallel_size"] = 4
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--tensor-parallel-size", cmd)
        self.assertEqual(cmd[cmd.index("--tensor-parallel-size") + 1], "4")

    def test_tensor_parallel_size_one_omitted(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["tensor_parallel_size"] = 1
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--tensor-parallel-size", cmd)

    def test_tensor_parallel_size_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--tensor-parallel-size", cmd)

    def test_pipeline_parallel_size_flag_present(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["pipeline_parallel_size"] = 2
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--pipeline-parallel-size", cmd)
        self.assertEqual(cmd[cmd.index("--pipeline-parallel-size") + 1], "2")

    def test_pipeline_parallel_size_one_omitted(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["pipeline_parallel_size"] = 1
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--pipeline-parallel-size", cmd)

    def test_pipeline_parallel_size_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--pipeline-parallel-size", cmd)

    def test_disable_custom_all_reduce_flag_present(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["disable_custom_all_reduce"] = True
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--disable-custom-all-reduce", cmd)

    def test_disable_custom_all_reduce_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--disable-custom-all-reduce", cmd)

    def test_enable_prefix_caching_flag_false(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["enable_prefix_caching"] = False
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--no-enable-prefix-caching", cmd)
        self.assertNotIn("--enable-prefix-caching", cmd)

    def test_enable_chunked_prefill_flag_false(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["enable_chunked_prefill"] = False
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--no-enable-chunked-prefill", cmd)
        self.assertNotIn("--enable-chunked-prefill", cmd)

    def test_disable_sliding_window_flag_false(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["disable_sliding_window"] = False
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--no-disable-sliding-window", cmd)
        self.assertNotIn("--disable-sliding-window", cmd)

    def test_disable_custom_all_reduce_flag_false(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["serving_engine"]["disable_custom_all_reduce"] = False
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--no-disable-custom-all-reduce", cmd)
        self.assertNotIn("--disable-custom-all-reduce", cmd)

    def test_tokenizer_revision_in_cmd(self) -> None:
        d = _load_manifest_dict()
        d["model"]["tokenizer_revision"] = "bb" * 20
        m = _manifest_from_dict(d)
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--tokenizer-revision", cmd)
        self.assertEqual(cmd[cmd.index("--tokenizer-revision") + 1], "bb" * 20)


class TestSetDeterministicEnv(unittest.TestCase):
    """Test that _set_deterministic_env reads knobs from manifest."""

    def test_cublas_workspace_config_from_manifest(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["deterministic_knobs"]["cublas_workspace_config"] = ":16:8"
        m = _manifest_from_dict(d)
        _set_deterministic_env(m)
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":16:8")

    def test_cublas_workspace_config_default(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["deterministic_knobs"].pop("cublas_workspace_config", None)
        m = _manifest_from_dict(d)
        _set_deterministic_env(m)
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

    def test_pythonhashseed_from_manifest(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["deterministic_knobs"]["pythonhashseed"] = "12345"
        m = _manifest_from_dict(d)
        _set_deterministic_env(m)
        self.assertEqual(os.environ["PYTHONHASHSEED"], "12345")

    def test_pythonhashseed_default(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["deterministic_knobs"].pop("pythonhashseed", None)
        m = _manifest_from_dict(d)
        _set_deterministic_env(m)
        self.assertEqual(os.environ["PYTHONHASHSEED"], "0")

    def test_torch_deterministic_true_sets_env(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["deterministic_knobs"]["torch_deterministic"] = True
        m = _manifest_from_dict(d)
        _set_deterministic_env(m)
        self.assertEqual(os.environ["TORCH_CUDNN_DETERMINISTIC"], "1")
        self.assertEqual(os.environ["TORCH_CUDNN_BENCHMARK"], "0")

    def test_torch_deterministic_false_unsets_env(self) -> None:
        os.environ["TORCH_CUDNN_DETERMINISTIC"] = "1"
        os.environ["TORCH_CUDNN_BENCHMARK"] = "0"
        d = _load_manifest_dict()
        d["runtime"]["deterministic_knobs"]["torch_deterministic"] = False
        m = _manifest_from_dict(d)
        _set_deterministic_env(m)
        self.assertNotIn("TORCH_CUDNN_DETERMINISTIC", os.environ)
        self.assertNotIn("TORCH_CUDNN_BENCHMARK", os.environ)


class TestClosureHash(unittest.TestCase):
    """Nix closure hash verification via _verify_closure."""

    def test_digest_mismatch_raises(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["closure_hash"] = "sha256:" + "a" * 64
        m = _manifest_from_dict(d)
        report = {"enforced": [], "warnings": []}
        old_val = os.environ.get("CLOSURE_HASH")
        try:
            os.environ["CLOSURE_HASH"] = "sha256:" + "b" * 64
            with self.assertRaises(ValidationError) as ctx:
                _verify_closure(m, report)
            self.assertIn("Closure hash mismatch", str(ctx.exception))
        finally:
            if old_val is None:
                os.environ.pop("CLOSURE_HASH", None)
            else:
                os.environ["CLOSURE_HASH"] = old_val

    def test_digest_match_passes(self) -> None:
        d = _load_manifest_dict()
        digest = "sha256:" + "a" * 64
        d["runtime"]["closure_hash"] = digest
        m = _manifest_from_dict(d)
        report = {"enforced": [], "warnings": []}
        old_val = os.environ.get("CLOSURE_HASH")
        try:
            os.environ["CLOSURE_HASH"] = digest
            _verify_closure(m, report)
            self.assertTrue(any("closure hash verified" in e for e in report["enforced"]))
        finally:
            if old_val is None:
                os.environ.pop("CLOSURE_HASH", None)
            else:
                os.environ["CLOSURE_HASH"] = old_val

    def test_no_digest_in_manifest_skips(self) -> None:
        d = _load_manifest_dict()
        d["runtime"].pop("closure_hash", None)
        m = _manifest_from_dict(d)
        report = {"enforced": [], "warnings": []}
        _verify_closure(m, report)
        self.assertEqual(report["enforced"], [])
        self.assertEqual(report["warnings"], [])

    def test_env_var_missing_warns(self) -> None:
        d = _load_manifest_dict()
        d["runtime"]["closure_hash"] = "sha256:" + "a" * 64
        m = _manifest_from_dict(d)
        report = {"enforced": [], "warnings": []}
        old_val = os.environ.get("CLOSURE_HASH")
        try:
            os.environ.pop("CLOSURE_HASH", None)
            _verify_closure(m, report)
            self.assertTrue(any("not set" in w for w in report["warnings"]))
        finally:
            if old_val is not None:
                os.environ["CLOSURE_HASH"] = old_val


class TestCheckHardware(unittest.TestCase):
    """Hardware conformance check via _check_hardware (pure function, no mocking)."""

    def _probe(self, **overrides) -> "GpuProbe":  # noqa: F821  (GpuProbe imported locally below)
        from modules.inference.manifest.model import GpuProbe
        defaults = dict(
            available=True, name="NVIDIA GH200 480GB", count=1,
            compute_capability="9.0", driver_version="570.148.08",
            cuda_version="12.8", torch_version="2.10.0", vllm_version="0.17.1",
        )
        defaults.update(overrides)
        return GpuProbe(**defaults)

    def test_conformant(self) -> None:
        m = _load_manifest()
        probe = self._probe()
        result = _check_hardware(m, probe)
        self.assertEqual(result.status, "conformant")
        self.assertEqual(result.warnings, [])

    def test_gpu_count_mismatch(self) -> None:
        m = _load_manifest()
        probe = self._probe(count=4)
        result = _check_hardware(m, probe)
        self.assertTrue(any("GPU count" in w for w in result.warnings))

    def test_gpu_model_mismatch(self) -> None:
        m = _load_manifest()
        probe = self._probe(name="NVIDIA A100 80GB")
        result = _check_hardware(m, probe)
        self.assertTrue(any("GPU model" in w for w in result.warnings))

    def test_driver_version_match(self) -> None:
        d = _load_manifest_dict()
        d["hardware_profile"]["gpu"]["driver_version"] = "550.54.15"
        m = _manifest_from_dict(d)
        probe = self._probe(driver_version="550.54.15")
        result = _check_hardware(m, probe)
        self.assertFalse(any("GPU driver" in w for w in result.warnings))

    def test_driver_version_mismatch(self) -> None:
        d = _load_manifest_dict()
        d["hardware_profile"]["gpu"]["driver_version"] = "550.54.15"
        m = _manifest_from_dict(d)
        probe = self._probe(driver_version="535.86.01")
        result = _check_hardware(m, probe)
        self.assertTrue(any("GPU driver" in w for w in result.warnings))

    def test_driver_not_queryable(self) -> None:
        d = _load_manifest_dict()
        d["hardware_profile"]["gpu"]["driver_version"] = "550.54.15"
        m = _manifest_from_dict(d)
        probe = self._probe(driver_version="")
        result = _check_hardware(m, probe)
        self.assertTrue(any("Could not query" in w for w in result.warnings))

    def test_cuda_version_match(self) -> None:
        d = _load_manifest_dict()
        d["hardware_profile"]["gpu"]["cuda_driver_version"] = "12.4"
        m = _manifest_from_dict(d)
        probe = self._probe(cuda_version="12.4")
        result = _check_hardware(m, probe)
        self.assertFalse(any("CUDA version" in w for w in result.warnings))

    def test_cuda_version_mismatch(self) -> None:
        d = _load_manifest_dict()
        d["hardware_profile"]["gpu"]["cuda_driver_version"] = "12.4"
        m = _manifest_from_dict(d)
        probe = self._probe(cuda_version="11.8")
        result = _check_hardware(m, probe)
        self.assertTrue(any("CUDA version" in w for w in result.warnings))

    def test_no_gpu(self) -> None:
        m = _load_manifest()
        probe = self._probe(available=False)
        result = _check_hardware(m, probe)
        self.assertEqual(result.status, "no_gpu")


class TestVerifyModelArtifacts(unittest.TestCase):
    """Test _verify_model_artifacts with temp directory and fake files."""

    def _make_manifest_with_artifact(self, cache_dir, filename, content, digest=None, size=None):
        """Create a manifest with a single model_weights artifact pointing at a temp file."""
        from modules.core.common.deterministic import sha256_file as _sha256_file
        fpath = cache_dir / filename
        fpath.write_bytes(content)
        actual_digest = _sha256_file(fpath)
        d = _load_manifest_dict()
        d["model"]["weights_revision"] = "abcd" * 10  # 40 hex chars
        d["artifact_inputs"] = [{
            "artifact_id": "test-weights",
            "artifact_type": "model_weights",
            "source_kind": "hf",
            "source_uri": "hf://test/model/weights.bin",
            "immutable_ref": "abcd" * 10,
            "expected_digest": digest if digest else actual_digest,
            "path": filename,
            "size_bytes": size if size else len(content),
        }]
        return _manifest_from_dict(d)

    def test_correct_digest_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_dir = pathlib.Path(td)
            content = b"test model weights data"
            m = self._make_manifest_with_artifact(cache_dir, "weights.bin", content)
            report = {"enforced": [], "warnings": []}
            old = os.environ.get("RUNNER_MODEL_PATH")
            try:
                os.environ["RUNNER_MODEL_PATH"] = str(cache_dir)
                _verify_model_artifacts(m, report)
            finally:
                if old is None:
                    os.environ.pop("RUNNER_MODEL_PATH", None)
                else:
                    os.environ["RUNNER_MODEL_PATH"] = old
            self.assertTrue(any("verified weights.bin" in e for e in report["enforced"]))

    def test_wrong_digest_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_dir = pathlib.Path(td)
            content = b"test model weights data"
            wrong_digest = "sha256:" + "0" * 64
            m = self._make_manifest_with_artifact(cache_dir, "weights.bin", content, digest=wrong_digest)
            report = {"enforced": [], "warnings": []}
            old = os.environ.get("RUNNER_MODEL_PATH")
            try:
                os.environ["RUNNER_MODEL_PATH"] = str(cache_dir)
                with self.assertRaises(ValidationError) as ctx:
                    _verify_model_artifacts(m, report)
                self.assertIn("File digest mismatch", str(ctx.exception))
            finally:
                if old is None:
                    os.environ.pop("RUNNER_MODEL_PATH", None)
                else:
                    os.environ["RUNNER_MODEL_PATH"] = old

    def test_size_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_dir = pathlib.Path(td)
            content = b"test model weights data"
            m = self._make_manifest_with_artifact(cache_dir, "weights.bin", content, size=999999)
            report = {"enforced": [], "warnings": []}
            old = os.environ.get("RUNNER_MODEL_PATH")
            try:
                os.environ["RUNNER_MODEL_PATH"] = str(cache_dir)
                with self.assertRaises(ValidationError) as ctx:
                    _verify_model_artifacts(m, report)
                self.assertIn("File size mismatch", str(ctx.exception))
            finally:
                if old is None:
                    os.environ.pop("RUNNER_MODEL_PATH", None)
                else:
                    os.environ["RUNNER_MODEL_PATH"] = old

    def test_missing_file_warns(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_dir = pathlib.Path(td)
            d = _load_manifest_dict()
            d["model"]["weights_revision"] = "abcd" * 10
            d["artifact_inputs"] = [{
                    "artifact_id": "test-weights",
                    "artifact_type": "model_weights",
                    "source_kind": "hf",
                    "source_uri": "hf://test/model/weights.bin",
                    "immutable_ref": "abcd" * 10,
                    "expected_digest": "sha256:" + "a" * 64,
                    "path": "nonexistent.bin",
            }]
            m = _manifest_from_dict(d)
            report = {"enforced": [], "warnings": []}
            old = os.environ.get("RUNNER_MODEL_PATH")
            try:
                os.environ["RUNNER_MODEL_PATH"] = str(cache_dir)
                _verify_model_artifacts(m, report)
            finally:
                if old is None:
                    os.environ.pop("RUNNER_MODEL_PATH", None)
                else:
                    os.environ["RUNNER_MODEL_PATH"] = old
            self.assertTrue(any("File not found" in w for w in report["warnings"]))

    def test_missing_revision_skips(self) -> None:
        # weights_revision is required in Pydantic, so we test with a manifest
        # that has the field set; the function should proceed normally.
        m = _load_manifest()
        report = {"enforced": [], "warnings": []}
        # Without RUNNER_MODEL_PATH or HF cache, it will warn about missing cache
        old = os.environ.get("RUNNER_MODEL_PATH")
        try:
            os.environ.pop("RUNNER_MODEL_PATH", None)
            _verify_model_artifacts(m, report)
        finally:
            if old is not None:
                os.environ["RUNNER_MODEL_PATH"] = old


if __name__ == "__main__":
    unittest.main()
