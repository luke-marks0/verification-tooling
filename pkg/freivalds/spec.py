"""Dataclasses for the Freivalds attestation protocol.

The wire format pairs a small JSON envelope (this module's `to_dict`/`from_dict`)
with a binary payload per matmul (the bytes of `C`). The JSON envelope is
canonicalised with `pkg.common.deterministic.canonical_json_text`; the bytes are
referenced by index and digested separately. See
``experiments/freivalds-attestation/plan.md`` §Protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# Supported dtype names. Stdlib backend handles {int8, int32, fp64};
# torch backend handles all of these.
SUPPORTED_DTYPES = frozenset({
    "int8", "int32",
    "fp16", "bf16", "fp32", "fp64",
    "fp8_e4m3",
})

# Dtypes whose values are exactly representable -> bitwise comparison.
INTEGER_DTYPES = frozenset({"int8", "int32"})

# Dtypes that produce float output -> tolerance comparison.
FLOAT_DTYPES = frozenset({"fp16", "bf16", "fp32", "fp64", "fp8_e4m3"})


class ComparisonMode(str, Enum):
    BITWISE = "bitwise"
    TOLERANCE = "tolerance"


@dataclass(frozen=True)
class Tolerance:
    """Float-comparison tolerance, used when ``ComparisonMode.TOLERANCE`` applies.

    Soundness statement: any divergence with
    ``|A(Br) - Cr|_inf > atol + rtol * |Cr|_inf`` is detected.
    """
    atol: float
    rtol: float

    def to_dict(self) -> dict[str, float]:
        return {"atol": float(self.atol), "rtol": float(self.rtol)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Tolerance:
        return cls(atol=float(d["atol"]), rtol=float(d["rtol"]))


@dataclass(frozen=True)
class MatmulSpec:
    """One matmul in a challenge.

    Shape: ``A: M x K`` at ``dtype_a``; ``B: K x N`` at ``dtype_b``;
    ``C = A @ B`` accumulated at ``dtype_acc`` and downcast to ``dtype_c``.
    Seeds expand into deterministic bytes via :mod:`pkg.freivalds.prng`.
    """
    id: str
    M: int
    K: int
    N: int
    dtype_a: str
    dtype_b: str
    dtype_acc: str
    dtype_c: str
    seed_a: int
    seed_b: int
    comparison: ComparisonMode
    tolerance: Tolerance | None = None

    def __post_init__(self) -> None:
        for name in ("dtype_a", "dtype_b", "dtype_acc", "dtype_c"):
            v = getattr(self, name)
            if v not in SUPPORTED_DTYPES:
                raise ValueError(f"unsupported {name}: {v!r}")
        if self.M < 1 or self.K < 1 or self.N < 1:
            raise ValueError(f"dims must be positive: M={self.M} K={self.K} N={self.N}")
        if self.comparison is ComparisonMode.BITWISE and self.dtype_c not in INTEGER_DTYPES:
            raise ValueError(
                f"bitwise comparison requires integer dtype_c, got {self.dtype_c}"
            )
        if self.comparison is ComparisonMode.TOLERANCE and self.tolerance is None:
            raise ValueError("tolerance comparison requires a Tolerance")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "M": self.M,
            "K": self.K,
            "N": self.N,
            "dtype_a": self.dtype_a,
            "dtype_b": self.dtype_b,
            "dtype_acc": self.dtype_acc,
            "dtype_c": self.dtype_c,
            "seed_a": int(self.seed_a),
            "seed_b": int(self.seed_b),
            "comparison": self.comparison.value,
        }
        if self.tolerance is not None:
            d["tolerance"] = self.tolerance.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MatmulSpec:
        tol = Tolerance.from_dict(d["tolerance"]) if "tolerance" in d else None
        return cls(
            id=str(d["id"]),
            M=int(d["M"]),
            K=int(d["K"]),
            N=int(d["N"]),
            dtype_a=str(d["dtype_a"]),
            dtype_b=str(d["dtype_b"]),
            dtype_acc=str(d["dtype_acc"]),
            dtype_c=str(d["dtype_c"]),
            seed_a=int(d["seed_a"]),
            seed_b=int(d["seed_b"]),
            comparison=ComparisonMode(d["comparison"]),
            tolerance=tol,
        )


@dataclass(frozen=True)
class Challenge:
    """A bundle of matmuls a verifier asks a prover to compute.

    ``matmuls_per_response`` is the streaming-protocol stride ``M`` (Buck's
    "matmuls per response" parameter from the 2026-04-30 protocol meeting).
    When ``None`` or equal to ``len(matmuls)``, the prover runs in single-shot
    mode and returns full ``C`` bytes (Freivalds verifier path). When set to
    a smaller value, the prover runs in streaming mode and returns one
    chain-hash per chunk of ``M`` consecutive matmuls — see
    ``experiments/freivalds-attestation/specs/streaming_strided.md``.
    """
    challenge_id: str
    matmuls: tuple[MatmulSpec, ...]
    matmuls_per_response: int | None = None

    def __post_init__(self) -> None:
        if self.matmuls_per_response is not None:
            M = int(self.matmuls_per_response)
            if M < 1:
                raise ValueError(f"matmuls_per_response must be >= 1, got {M}")
            if M > len(self.matmuls):
                raise ValueError(
                    f"matmuls_per_response={M} > total matmuls={len(self.matmuls)}"
                )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "challenge_version": "v1",
            "challenge_id": self.challenge_id,
            "matmuls": [m.to_dict() for m in self.matmuls],
        }
        if self.matmuls_per_response is not None:
            d["matmuls_per_response"] = int(self.matmuls_per_response)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Challenge:
        if d.get("challenge_version") != "v1":
            raise ValueError(f"unsupported challenge_version: {d.get('challenge_version')!r}")
        ids = [m["id"] for m in d["matmuls"]]
        if len(set(ids)) != len(ids):
            raise ValueError("matmul ids must be unique within a challenge")
        mpr = d.get("matmuls_per_response")
        return cls(
            challenge_id=str(d["challenge_id"]),
            matmuls=tuple(MatmulSpec.from_dict(m) for m in d["matmuls"]),
            matmuls_per_response=int(mpr) if mpr is not None else None,
        )


@dataclass(frozen=True)
class MatmulResult:
    """Prover's answer for one matmul.

    ``c_b64`` is the base64-encoded canonical row-major byte representation
    of ``C`` at ``dtype_c``. ``digest_a``/``digest_b``/``digest_c`` are
    sha256 of the same canonical bytes for each matrix.
    Timing fields are observed but do not affect the verdict in v1.
    """
    id: str
    digest_a: str
    digest_b: str
    digest_c: str
    c_b64: str
    wall_time_ms: float
    device: str = ""
    device_name: str = ""
    nvml_clock_mhz: int | None = None
    nvml_temp_c: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "digest_a": self.digest_a,
            "digest_b": self.digest_b,
            "digest_c": self.digest_c,
            "c_b64": self.c_b64,
            "wall_time_ms": float(self.wall_time_ms),
            "device": self.device,
            "device_name": self.device_name,
        }
        if self.nvml_clock_mhz is not None:
            d["nvml_clock_mhz"] = int(self.nvml_clock_mhz)
        if self.nvml_temp_c is not None:
            d["nvml_temp_c"] = int(self.nvml_temp_c)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MatmulResult:
        return cls(
            id=str(d["id"]),
            digest_a=str(d["digest_a"]),
            digest_b=str(d["digest_b"]),
            digest_c=str(d["digest_c"]),
            c_b64=str(d["c_b64"]),
            wall_time_ms=float(d["wall_time_ms"]),
            device=str(d.get("device", "")),
            device_name=str(d.get("device_name", "")),
            nvml_clock_mhz=int(d["nvml_clock_mhz"]) if "nvml_clock_mhz" in d else None,
            nvml_temp_c=int(d["nvml_temp_c"]) if "nvml_temp_c" in d else None,
        )


@dataclass(frozen=True)
class ChainHashChunk:
    """One chunk of the streaming/strided protocol response.

    Per Luke's 2026-04-30 design, the prover hash-chains the ``digest_c``
    values of ``M`` consecutive matmuls into one chain hash and reports it
    in lieu of returning full C bytes. The verifier checks the chain bound
    by re-running the same matmuls (or a subset) against the same seeds.
    """
    chunk_index: int
    matmul_ids: tuple[str, ...]
    chain_hash: str
    wall_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": int(self.chunk_index),
            "matmul_ids": list(self.matmul_ids),
            "chain_hash": self.chain_hash,
            "wall_time_ms": float(self.wall_time_ms),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChainHashChunk:
        return cls(
            chunk_index=int(d["chunk_index"]),
            matmul_ids=tuple(str(x) for x in d["matmul_ids"]),
            chain_hash=str(d["chain_hash"]),
            wall_time_ms=float(d.get("wall_time_ms", 0.0)),
        )


@dataclass(frozen=True)
class Response:
    """Prover's response to a challenge.

    ``results`` carries full per-matmul results (single-shot mode);
    ``chain_hashes`` carries one chunk-hash per stride-M chunk (streaming
    mode, see :class:`ChainHashChunk`). Exactly one of them is populated
    in normal use, but both are wire-compatible with v1 verifiers (which
    will simply ignore the empty tuple).
    """
    challenge_id: str
    backend: str
    results: tuple[MatmulResult, ...]
    chain_hashes: tuple[ChainHashChunk, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "response_version": "v1",
            "challenge_id": self.challenge_id,
            "backend": self.backend,
            "results": [r.to_dict() for r in self.results],
        }
        if self.chain_hashes:
            d["chain_hashes"] = [c.to_dict() for c in self.chain_hashes]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Response:
        if d.get("response_version") != "v1":
            raise ValueError(f"unsupported response_version: {d.get('response_version')!r}")
        chain_raw = d.get("chain_hashes") or []
        return cls(
            challenge_id=str(d["challenge_id"]),
            backend=str(d["backend"]),
            results=tuple(MatmulResult.from_dict(r) for r in d["results"]),
            chain_hashes=tuple(ChainHashChunk.from_dict(c) for c in chain_raw),
        )


@dataclass(frozen=True)
class MatmulVerdict:
    """Per-matmul verifier verdict."""
    id: str
    passed: bool
    reason: str
    max_abs_diff: float
    cr_inf_norm: float
    wall_time_ms: float
    digest_a_match: bool
    digest_b_match: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "passed": bool(self.passed),
            "reason": self.reason,
            "max_abs_diff": float(self.max_abs_diff),
            "cr_inf_norm": float(self.cr_inf_norm),
            "wall_time_ms": float(self.wall_time_ms),
            "digest_a_match": bool(self.digest_a_match),
            "digest_b_match": bool(self.digest_b_match),
        }


@dataclass(frozen=True)
class AttestationReport:
    """Verifier's report over a challenge response."""
    challenge_id: str
    backend: str
    overall_passed: bool
    matmuls: tuple[MatmulVerdict, ...] = field(default_factory=tuple)
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "attestation_version": "v1",
            "challenge_id": self.challenge_id,
            "backend": self.backend,
            "overall_passed": bool(self.overall_passed),
            "matmuls": [m.to_dict() for m in self.matmuls],
            "generated_at": self.generated_at,
        }
