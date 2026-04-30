"""Streaming/strided proof-of-compute protocol (Luke's 2026-04-30 design).

This is the M-stride extension of the single-shot Freivalds protocol. The
verifier picks ``K`` total matmuls and a stride ``M < K``; the prover hashes
the per-matmul ``digest_c`` of every batch of ``M`` consecutive matmuls into
one chain hash and reports it. The protocol bounds how much external help
the prover can fetch — see
``experiments/freivalds-attestation/specs/streaming_strided.md``.

This module implements the prover and verifier sides of the chain-hash
construction. It deliberately reuses the existing per-matmul kernels and
PRNG so that a streaming response is just a re-aggregation of what the
single-shot mode already produces — i.e., a streaming-mode response is
verifiable without the prover ever sending C bytes.

Wire-format compatibility: a streaming-mode :class:`Response` carries an
empty ``results`` tuple and a non-empty ``chain_hashes`` tuple. v1
verifiers ignore ``chain_hashes`` and will return an empty (no-op) report
because no per-matmul results are present; v2 verifiers branch on
``Challenge.matmuls_per_response``.
"""
from __future__ import annotations

import hashlib

from pkg.common.deterministic import utc_now_iso
from pkg.freivalds import prng
from pkg.freivalds.spec import (
    AttestationReport,
    ChainHashChunk,
    Challenge,
    MatmulVerdict,
    Response,
)


# Genesis hash for the chain. Distinct domain-separation tag so an attacker
# can't replay a hash from another protocol layer (e.g., matrix_digest).
GENESIS_CHAIN_HASH = "sha256:" + hashlib.sha256(
    b"freivalds-streaming-chain-v1|genesis"
).hexdigest()


def fold_chain_hash(prev: str, digest_c: str) -> str:
    """Fold one matmul's ``digest_c`` into the chain.

    ``prev`` and ``digest_c`` are the canonical ``sha256:<hex>`` strings the
    rest of the protocol uses. The chain is ``H(prev_bytes || digest_bytes)``
    so a single misordered or substituted matmul perturbs every downstream
    chain hash — there is no commutativity to exploit.
    """
    h = hashlib.sha256()
    h.update(prev.encode("ascii"))
    h.update(b"|")
    h.update(digest_c.encode("ascii"))
    return f"sha256:{h.hexdigest()}"


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield i, seq[i:i + size]


def execute_streaming_challenge(challenge: Challenge, backend) -> Response:
    """Run a streaming-mode challenge.

    Requires ``challenge.matmuls_per_response`` to be set. The prover
    materialises (A, B, C) per matmul exactly as in single-shot mode but
    discards the C bytes after extracting ``digest_c``, then folds digests
    into a per-chunk chain hash.
    """
    M = challenge.matmuls_per_response
    if M is None:
        raise ValueError(
            "execute_streaming_challenge requires matmuls_per_response to be set; "
            "use execute_challenge() for single-shot mode"
        )

    chunks: list[ChainHashChunk] = []
    info = backend.device_info()
    for chunk_idx, batch in _chunks(challenge.matmuls, M):
        chain = GENESIS_CHAIN_HASH
        ids: list[str] = []
        chunk_wall_ms = 0.0
        for spec in batch:
            A, A_bytes = backend.gen_matrix(spec.seed_a, spec.dtype_a, spec.M, spec.K)
            B, B_bytes = backend.gen_matrix(spec.seed_b, spec.dtype_b, spec.K, spec.N)
            t0 = backend.perf_time_ms()
            C = backend.matmul(A, B, spec.dtype_a, spec.dtype_b,
                               spec.dtype_acc, spec.dtype_c)
            t1 = backend.perf_time_ms()
            chunk_wall_ms += (t1 - t0)
            C_bytes = backend.write_matrix_to_bytes(C, spec.dtype_c)
            digest_c = prng.matrix_digest(C_bytes)
            chain = fold_chain_hash(chain, digest_c)
            ids.append(spec.id)
        chunks.append(ChainHashChunk(
            chunk_index=chunk_idx // M,
            matmul_ids=tuple(ids),
            chain_hash=chain,
            wall_time_ms=chunk_wall_ms,
        ))

    return Response(
        challenge_id=challenge.challenge_id,
        backend=backend.name,
        results=(),
        chain_hashes=tuple(chunks),
    )


def verify_streaming_response(
    challenge: Challenge,
    response: Response,
    backend,
) -> AttestationReport:
    """Verify a streaming-mode response by recomputing each chain locally.

    The verifier re-runs every matmul in the challenge against the same
    seeds, folds the resulting ``digest_c`` values into the same chain
    construction, and compares chunk-by-chunk against the prover's reported
    chain hashes. A mismatch on chunk *k* means at least one of the matmuls
    in that chunk produced a different ``C`` (or the prover cheated on the
    chain order).

    NOTE: this is the audit/correctness check. In production, the verifier
    only redoes a sampled subset of chunks (untyped here) — but the
    interface is the same.
    """
    if response.challenge_id != challenge.challenge_id:
        return AttestationReport(
            challenge_id=challenge.challenge_id,
            backend=response.backend,
            overall_passed=False,
            matmuls=tuple(),
            generated_at=utc_now_iso(),
        )
    M = challenge.matmuls_per_response
    if M is None:
        raise ValueError("verify_streaming_response requires matmuls_per_response")

    expected = list(response.chain_hashes)
    by_chunk_index = {c.chunk_index: c for c in expected}

    verdicts: list[MatmulVerdict] = []
    overall = True
    for chunk_idx, batch in _chunks(challenge.matmuls, M):
        idx = chunk_idx // M
        prover_chunk = by_chunk_index.get(idx)
        chain = GENESIS_CHAIN_HASH
        for spec in batch:
            A, A_bytes = backend.gen_matrix(spec.seed_a, spec.dtype_a, spec.M, spec.K)
            B, B_bytes = backend.gen_matrix(spec.seed_b, spec.dtype_b, spec.K, spec.N)
            C = backend.matmul(A, B, spec.dtype_a, spec.dtype_b,
                               spec.dtype_acc, spec.dtype_c)
            C_bytes = backend.write_matrix_to_bytes(C, spec.dtype_c)
            chain = fold_chain_hash(chain, prng.matrix_digest(C_bytes))

        if prover_chunk is None:
            overall = False
            for spec in batch:
                verdicts.append(MatmulVerdict(
                    id=spec.id, passed=False,
                    reason=f"missing chain_hash for chunk {idx}",
                    max_abs_diff=0.0, cr_inf_norm=0.0, wall_time_ms=0.0,
                    digest_a_match=False, digest_b_match=False,
                ))
            continue

        passed = (chain == prover_chunk.chain_hash)
        if not passed:
            overall = False
        reason = "chain hash matches" if passed else (
            f"chain hash mismatch on chunk {idx}: "
            f"prover={prover_chunk.chain_hash}, verifier={chain}"
        )
        # One verdict per matmul in the chunk; per-matmul granularity is
        # not available from a chain hash, so all matmuls in a failing
        # chunk fail collectively.
        for spec in batch:
            verdicts.append(MatmulVerdict(
                id=spec.id, passed=passed, reason=reason,
                max_abs_diff=0.0, cr_inf_norm=0.0,
                wall_time_ms=prover_chunk.wall_time_ms / max(1, len(batch)),
                digest_a_match=True, digest_b_match=True,
            ))

    return AttestationReport(
        challenge_id=challenge.challenge_id,
        backend=response.backend,
        overall_passed=overall,
        matmuls=tuple(verdicts),
        generated_at=utc_now_iso(),
    )
