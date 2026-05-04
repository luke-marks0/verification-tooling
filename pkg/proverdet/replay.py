"""Replay-evidence builders for the prover.

`produce_evidence_stream` runs a Freivalds challenge for each round of the
replay request's proof_of_work spec, stashes each per-matmul attestation
in the provided store, and yields a sequence of NDJSON-shaped chunks: one
`{"kind": "pow", ...}` per round followed by a final
`{"kind": "evidence", ...full ReplayEvidence}`. The `/replay` endpoint
streams these chunks over an `application/x-ndjson` response (Task 6.3).

`produce_evidence` is the sync wrapper; it consumes the stream and returns
the final `ReplayEvidence`. Used by tests + Task 6.4's verifier-side
verification.

The wire dtype (`bf16` | `fp16` | `int8`) maps onto a freivalds MatmulSpec
dtype combo. Stdlib backend supports only the `int8` mapping (with int32
accumulator + int32 output, bitwise comparison); the torch backend adds
`bf16` and `fp16`. Tests use `int8` + small `matmul_dim` to stay CPU-only.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from pkg.common.deterministic import sha256_prefixed, utc_now_iso
from pkg.freivalds import (
    Challenge,
    ComparisonMode,
    MatmulSpec,
    Tolerance,
    execute_challenge,
)
from pkg.proverdet.attestation_store import AttestationStore
from pkg.proverdet.erasure import HmacErasureBackend, run_erasure
from pkg.proverdet.wire import (
    PowStreamEntry,
    ReplayEvidence,
    ReplayOutput,
    ReplayRequest,
)


class FreivaldsBackend(Protocol):
    """Subset of the backend interface produce_evidence needs."""

    name: str

    def device_info(self) -> dict[str, Any]: ...
    def perf_time_ms(self) -> float: ...
    def gen_matrix(self, seed: int, dtype: str, rows: int, cols: int) -> tuple[Any, bytes]: ...
    def matmul(
        self,
        A: Any,
        B: Any,
        dtype_a: str,
        dtype_b: str,
        dtype_acc: str,
        dtype_c: str,
    ) -> Any: ...
    def write_matrix_to_bytes(self, matrix: Any, dtype: str) -> bytes: ...


# Wire dtype → MatmulSpec dtype combo + comparison mode + tolerance.
# int8: integer arithmetic, bitwise check.
# bf16/fp16: float arithmetic, tolerance check (atol/rtol generous for
# half-precision).
_DTYPE_COMBOS: dict[str, tuple[str, str, str, str, ComparisonMode, Tolerance | None]] = {
    "int8": ("int8", "int8", "int32", "int32", ComparisonMode.BITWISE, None),
    "fp16": (
        "fp16",
        "fp16",
        "fp32",
        "fp16",
        ComparisonMode.TOLERANCE,
        Tolerance(atol=1e-2, rtol=1e-2),
    ),
    "bf16": (
        "bf16",
        "bf16",
        "fp32",
        "bf16",
        ComparisonMode.TOLERANCE,
        Tolerance(atol=1e-1, rtol=1e-2),
    ),
}


def _seed_for(replay_id: str, matmul_id: str, role: str) -> int:
    """Deterministic 63-bit seed derived from (replay_id, matmul_id, role)."""
    h = hashlib.sha256(f"{replay_id}|{matmul_id}|{role}".encode()).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def _attestation_id(replay_id: str, matmul_id: str) -> str:
    h = hashlib.sha256(f"attest|{replay_id}|{matmul_id}".encode()).hexdigest()
    return f"att-{h[:32]}"


def check_supported(req: ReplayRequest) -> None:
    """Pre-flight check called before /replay opens its stream."""
    if req.proof_of_work.dtype not in _DTYPE_COMBOS:
        raise ValueError(f"unsupported proof_of_work.dtype: {req.proof_of_work.dtype!r}")


def produce_evidence_stream(
    req: ReplayRequest,
    *,
    freivalds_backend: FreivaldsBackend,
    attestation_store: AttestationStore,
    erasure_log_dir: Path,
) -> Iterator[dict[str, Any]]:
    """Run PoW + erasure and yield NDJSON-shaped chunks (Task 6.3 wire format).

    Yields, in order:
      * `{"kind": "pow", **PowStreamEntry}` once per round, as each
        Freivalds matmul completes.
      * Exactly one final `{"kind": "evidence", **ReplayEvidence}`.
    """
    check_supported(req)
    pow_spec = req.proof_of_work
    dtype_a, dtype_b, dtype_acc, dtype_c, comparison, tolerance = _DTYPE_COMBOS[pow_spec.dtype]

    pow_stream: list[PowStreamEntry] = []
    cumulative_t_ms = 0
    concatenated_c = bytearray()
    for i in range(pow_spec.rounds):
        matmul_id = f"m-{i:04d}"
        spec = MatmulSpec(
            id=matmul_id,
            M=pow_spec.matmul_dim,
            K=pow_spec.matmul_dim,
            N=pow_spec.matmul_dim,
            dtype_a=dtype_a,
            dtype_b=dtype_b,
            dtype_acc=dtype_acc,
            dtype_c=dtype_c,
            seed_a=_seed_for(req.replay_id, matmul_id, "a"),
            seed_b=_seed_for(req.replay_id, matmul_id, "b"),
            comparison=comparison,
            tolerance=tolerance,
        )
        # Run one matmul at a time so we can emit a pow chunk as it
        # completes — this is what makes the response a stream rather
        # than a single batched response.
        single_challenge = Challenge(challenge_id=f"chal-{req.replay_id}", matmuls=(spec,))
        single_response = execute_challenge(single_challenge, freivalds_backend)
        result = single_response.results[0]

        attestation_id = _attestation_id(req.replay_id, spec.id)
        attestation_store.put(
            attestation_id,
            {
                "matmul_id": spec.id,
                "challenge": single_challenge.to_dict(),
                "response": single_response.to_dict(),
            },
        )

        cumulative_t_ms += max(0, int(result.wall_time_ms))
        entry = PowStreamEntry(
            t_ms=cumulative_t_ms,
            freivalds_attestation_id=attestation_id,
            matmul_dim=pow_spec.matmul_dim,
            rounds=1,
            dtype=pow_spec.dtype,
        )
        pow_stream.append(entry)
        concatenated_c.extend(base64.b64decode(result.c_b64.encode("ascii")))
        yield {"kind": "pow", **entry.model_dump()}

    output_bytes = bytes(concatenated_c)
    commitment = sha256_prefixed(output_bytes)

    erasure_log_path = erasure_log_dir / f"erasure-{req.replay_id}.jsonl"
    erasure_evidence = run_erasure(
        req.erasure,
        log_path=erasure_log_path,
        backend=HmacErasureBackend(),
    )

    evidence = ReplayEvidence(
        replay_id=req.replay_id,
        produced_at=utc_now_iso(),
        output=ReplayOutput(
            commitment=commitment,
            bytes_b64=base64.b64encode(output_bytes).decode("ascii"),
        ),
        erasure_evidence=erasure_evidence,
        pow_stream=pow_stream,
    )
    yield {"kind": "evidence", **evidence.model_dump(exclude_none=True)}


def produce_evidence(
    req: ReplayRequest,
    *,
    freivalds_backend: FreivaldsBackend,
    attestation_store: AttestationStore,
    erasure_log_dir: Path,
) -> ReplayEvidence:
    """Sync wrapper over produce_evidence_stream — returns the final evidence."""
    final: dict[str, Any] | None = None
    for chunk in produce_evidence_stream(
        req,
        freivalds_backend=freivalds_backend,
        attestation_store=attestation_store,
        erasure_log_dir=erasure_log_dir,
    ):
        if chunk["kind"] == "evidence":
            final = chunk
    if final is None:
        raise RuntimeError("produce_evidence_stream did not yield an evidence chunk")
    body = {k: v for k, v in final.items() if k != "kind"}
    return ReplayEvidence.model_validate(body)
