"""Correctness & integrity attestation — stable public API. See ``README.md``.

Three primitives, re-exported as the curated surface:
  * matmul attestation via Freivalds' algorithm (``pkg.freivalds``)
  * token-level commitments (``pkg.e2e``)
  * run-bundle comparison / verdict (``cmd.verifier`` via ``modules._cmd``)
"""
from __future__ import annotations

from typing import Any

from pkg.e2e import commit_token, commit_token_stream
from pkg.freivalds import (
    AttestationReport,
    Challenge,
    ComparisonMode,
    MatmulSpec,
    Response,
    Tolerance,
    execute_challenge,
    verify_response,
)

from modules._cmd import verify_runs

__all__ = [
    # matmul attestation
    "Challenge",
    "MatmulSpec",
    "Response",
    "AttestationReport",
    "ComparisonMode",
    "Tolerance",
    "execute_challenge",
    "verify_response",
    "attest_matmuls",
    # token commitments
    "commit_token",
    "commit_token_stream",
    # run comparison
    "verify_runs",
]


def attest_matmuls(challenge: Challenge, backend: Any | None = None) -> AttestationReport:
    """Honest prover -> verifier round-trip for a matmul challenge.

    Convenience wrapper: executes ``challenge`` on ``backend`` (default: the
    pure-Python ``StdlibBackend``, no GPU) and verifies the response, returning
    the :class:`AttestationReport` (``.overall_passed`` is the verdict).
    """
    if backend is None:
        from pkg.freivalds.backends.stdlib import StdlibBackend

        backend = StdlibBackend()
    response = execute_challenge(challenge, backend)
    return verify_response(challenge, response, backend)
