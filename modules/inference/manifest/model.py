"""Pydantic models for the deterministic serving manifest (v1).

This is the single source of truth for manifest field names and types.
The JSON Schema in schemas/manifest.v1.schema.json is hand-maintained
separately; both must agree on structure.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# -- Enums --


class AttentionBackend(str, Enum):
    FLASH_ATTN = "FLASH_ATTN"
    TRITON_ATTN = "TRITON_ATTN"
    FLASH_ATTN_MLA = "FLASH_ATTN_MLA"
    TRITON_MLA = "TRITON_MLA"


class Dtype(str, Enum):
    auto = "auto"
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"


class ComparisonMode(str, Enum):
    exact = "exact"
    ulp = "ulp"
    absrel = "absrel"
    hash = "hash"


class ArtifactType(str, Enum):
    model_weights = "model_weights"
    model_config = "model_config"
    tokenizer = "tokenizer"
    generation_config = "generation_config"
    chat_template = "chat_template"
    prompt_formatter = "prompt_formatter"
    serving_stack = "serving_stack"
    container_image = "container_image"
    cuda_lib = "cuda_lib"
    kernel_library = "kernel_library"
    network_stack_binary = "network_stack_binary"
    pmd_driver = "pmd_driver"
    runtime_knob_set = "runtime_knob_set"
    request_set = "request_set"
    batching_policy = "batching_policy"
    nic_link_config = "nic_link_config"
    collective_stack = "collective_stack"
    compiled_extension = "compiled_extension"
    remote_code = "remote_code"


class SourceKind(str, Enum):
    hf = "hf"
    oci = "oci"
    s3 = "s3"
    http = "http"
    git = "git"
    nix = "nix"
    inline = "inline"


class FileRole(str, Enum):
    weights_shard = "weights_shard"
    config = "config"
    tokenizer = "tokenizer"
    generation_config = "generation_config"
    chat_template = "chat_template"
    prompt_formatter = "prompt_formatter"


class SchedulingPolicy(str, Enum):
    fcfs = "fcfs"
    priority = "priority"


class Quantization(str, Enum):
    awq = "awq"
    gptq = "gptq"
    bitsandbytes = "bitsandbytes"
    fp8 = "fp8"


class LoadFormat(str, Enum):
    auto = "auto"
    safetensors = "safetensors"
    pt = "pt"
    gguf = "gguf"


class KVCacheDtype(str, Enum):
    auto = "auto"
    fp8 = "fp8"
    int8 = "int8"


# -- Sub-models --


class GpuProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    count: int = Field(ge=1)
    driver_version: str = Field(min_length=1)
    cuda_driver_version: str = Field(min_length=1)


class HardwareProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu: GpuProfile


class ServingEngine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_model_len: int = Field(ge=1)
    max_num_seqs: int = Field(ge=1)
    gpu_memory_utilization: float = Field(ge=0.1, le=1.0)
    dtype: Dtype = Dtype.auto
    attention_backend: AttentionBackend
    # Optional fields -- when absent, vLLM uses its own defaults
    quantization: Quantization | None = None
    load_format: LoadFormat | None = None
    kv_cache_dtype: KVCacheDtype | None = None
    max_num_batched_tokens: int | None = Field(default=None, ge=1)
    block_size: int | None = Field(default=None, ge=1)
    enable_prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    scheduling_policy: SchedulingPolicy | None = None
    disable_sliding_window: bool | None = None
    tensor_parallel_size: int | None = Field(default=None, ge=1)
    pipeline_parallel_size: int | None = Field(default=None, ge=1)
    disable_custom_all_reduce: bool | None = None
    distributed_executor_backend: str | None = None


class BatchInvariance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    enforce_eager: bool


class DeterministicKnobs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0)
    torch_deterministic: bool
    cuda_launch_blocking: bool
    cublas_workspace_config: str | None = None
    pythonhashseed: str | None = None


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strict_hardware: bool
    batch_invariance: BatchInvariance
    deterministic_knobs: DeterministicKnobs
    serving_engine: ServingEngine
    closure_hash: str | None = Field(
        default=None, pattern=r"^sha256:[a-f0-9]{64}$"
    )


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(pattern=r"^hf://[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
    weights_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    tokenizer_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    trust_remote_code: bool


class RequestItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9._:-]+$")
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0, le=2)


class Comparator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ComparisonMode
    algorithm: str | None = None   # for hash mode
    ulp: int | None = None         # for ulp mode
    atol: float | None = None      # for absrel mode
    rtol: float | None = None      # for absrel mode

    @model_validator(mode="after")
    def _check_mode_fields(self) -> Comparator:
        if self.mode == ComparisonMode.hash and self.algorithm is None:
            raise ValueError("algorithm is required when mode is 'hash'")
        if self.mode == ComparisonMode.ulp and self.ulp is None:
            raise ValueError("ulp is required when mode is 'ulp'")
        if self.mode == ComparisonMode.absrel:
            if self.atol is None or self.rtol is None:
                raise ValueError("atol and rtol are required when mode is 'absrel'")
        return self


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens: Comparator
    logits: Comparator
    network_egress: Comparator | None = None


class ArtifactInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(pattern=r"^[A-Za-z0-9._:-]+$")
    artifact_type: ArtifactType
    source_kind: SourceKind
    source_uri: str
    immutable_ref: str = Field(min_length=1)
    # Optional fields
    name: str | None = None
    expected_digest: str | None = Field(
        default=None, pattern=r"^sha256:[a-f0-9]{64}$"
    )
    size_bytes: int | None = Field(default=None, ge=1)
    path: str | None = Field(default=None, min_length=1)
    role: FileRole | None = None


# -- Audit / replay verification --


class TokenCommitmentConfig(BaseModel):
    """Per-token commitment scheme used for the e2e replay audit loop.

    key_source "inline-shared" uses the hardcoded key in modules.attestation.e2e.crypto —
    it proves deterministic replay works but does NOT bind against a
    malicious provider. The label exists so future work can add
    "auditor-supplied" without another schema migration.
    """
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    algorithm: Literal["hmac-sha256"] = "hmac-sha256"
    key_source: Literal["inline-shared"] = "inline-shared"


class AuditConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_commitment: TokenCommitmentConfig


# -- Hardware conformance --


class GpuProbe(BaseModel):
    """Observed GPU state from the runtime environment."""
    available: bool = False
    name: str = ""
    count: int = 0
    compute_capability: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    torch_version: str = ""
    vllm_version: str = ""


class HardwareConformance(BaseModel):
    """Result of comparing manifest hardware_profile against actual hardware."""
    status: str  # "conformant", "no_gpu", "torch_not_available", "mismatch"
    probe: GpuProbe
    warnings: list[str] = Field(default_factory=list)


# -- Top-level manifest --


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_version: Literal["v1"] = "v1"
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    created_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
    model: ModelConfig
    runtime: RuntimeConfig
    hardware_profile: HardwareProfile
    requests: list[RequestItem] = Field(min_length=1)
    comparison: ComparisonConfig
    artifact_inputs: list[ArtifactInput] = Field(default_factory=list)
    audit: AuditConfig | None = None
