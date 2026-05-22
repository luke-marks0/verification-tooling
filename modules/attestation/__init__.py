"""Attestation capability — matmul / token / replay verification."""
from modules.attestation.api import (
    AttestationReport,
    Challenge,
    ComparisonMode,
    MatmulSpec,
    Response,
    Tolerance,
    attest_matmuls,
    commit_token,
    commit_token_stream,
    execute_challenge,
    verify_response,
    verify_runs,
)

__all__ = [
    "Challenge",
    "MatmulSpec",
    "Response",
    "AttestationReport",
    "ComparisonMode",
    "Tolerance",
    "execute_challenge",
    "verify_response",
    "attest_matmuls",
    "commit_token",
    "commit_token_stream",
    "verify_runs",
]
