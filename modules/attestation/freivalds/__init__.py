"""Freivalds matmul attestation primitives.

A verifier issues randomized matmul challenges; a prover executes them on
GPU and returns the results; the verifier checks correctness in O(n^2)
per matmul via Freivalds' algorithm. See
``experiments/freivalds-attestation/plan.md`` for the design.

Public API:

    from modules.attestation.freivalds import (
        Challenge, MatmulSpec, Response, MatmulResult, AttestationReport,
        execute_challenge, verify_response,
    )
"""
from __future__ import annotations

from modules.attestation.freivalds.spec import (
    AttestationReport,
    ChainHashChunk,
    Challenge,
    ComparisonMode,
    MatmulResult,
    MatmulSpec,
    MatmulVerdict,
    Response,
    Tolerance,
)
from modules.attestation.freivalds.prover import execute_challenge
from modules.attestation.freivalds.streaming import (
    GENESIS_CHAIN_HASH,
    execute_streaming_challenge,
    fold_chain_hash,
    verify_streaming_response,
)
from modules.attestation.freivalds.verifier import verify_response

__all__ = [
    "AttestationReport",
    "ChainHashChunk",
    "Challenge",
    "ComparisonMode",
    "GENESIS_CHAIN_HASH",
    "MatmulResult",
    "MatmulSpec",
    "MatmulVerdict",
    "Response",
    "Tolerance",
    "execute_challenge",
    "execute_streaming_challenge",
    "fold_chain_hash",
    "verify_response",
    "verify_streaming_response",
]
