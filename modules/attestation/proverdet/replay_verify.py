"""Verifier-side validation of a ReplayEvidence.

`verify_evidence(req, ev, fetch_attestation, backend)` returns a
VerdictResult of `(passed, reasons)`. A verdict is "passed" iff every
check returns no reason. Independent reasons stack — we don't short-
circuit, so the verifier sees every signal at once.

Checks:
  * Erasure: ev.erasure_evidence.rounds == req.erasure.rounds
    AND erasure_evidence.passed == erasure_evidence.rounds.
  * Cadence: len(ev.pow_stream) == req.proof_of_work.rounds.
  * Output commitment: sha256(decode(ev.output.bytes_b64)) ==
    ev.output.commitment.
  * Per-round Freivalds: for each pow entry, fetch the stored attestation
    and rerun verify_response with the C bytes extracted from
    ev.output.bytes_b64. This is what catches tampered output bytes — the
    attestations themselves are still honest, but the bytes the prover
    claims to have produced no longer satisfy the matmul.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from modules.core.common.deterministic import sha256_prefixed
from modules.attestation.freivalds import (
    Challenge,
    MatmulResult,
    Response,
    verify_response,
)
from modules.attestation.freivalds import prng as freivalds_prng
from modules.attestation.proverdet.replay import FreivaldsBackend
from modules.attestation.proverdet.wire import ReplayEvidence, ReplayRequest


@dataclass(frozen=True)
class VerdictResult:
    passed: bool
    reasons: list[str]


AttestationFetcher = Callable[[str], dict[str, Any] | None]


def verify_evidence(
    req: ReplayRequest,
    ev: ReplayEvidence,
    *,
    fetch_attestation: AttestationFetcher,
    backend: FreivaldsBackend,
) -> VerdictResult:
    reasons: list[str] = []

    # --- Erasure ---
    if ev.erasure_evidence.rounds != req.erasure.rounds:
        reasons.append(
            f"erasure rounds mismatch: evidence={ev.erasure_evidence.rounds} "
            f"vs request={req.erasure.rounds}"
        )
    if ev.erasure_evidence.passed < ev.erasure_evidence.rounds:
        reasons.append(
            f"erasure: only {ev.erasure_evidence.passed}/{ev.erasure_evidence.rounds} rounds passed"
        )

    # --- Cadence ---
    if len(ev.pow_stream) != req.proof_of_work.rounds:
        reasons.append(
            f"cadence: pow_stream length {len(ev.pow_stream)} != "
            f"requested rounds {req.proof_of_work.rounds}"
        )

    # --- Output bytes ---
    output_bytes = base64.b64decode(ev.output.bytes_b64)
    if sha256_prefixed(output_bytes) != ev.output.commitment:
        reasons.append("output commitment does not match bytes_b64")

    # --- Per-round Freivalds ---
    cursor = 0
    for entry in ev.pow_stream:
        att = fetch_attestation(entry.freivalds_attestation_id)
        if att is None:
            reasons.append(f"Freivalds: attestation {entry.freivalds_attestation_id} not found")
            continue
        try:
            challenge = Challenge.from_dict(att["challenge"])
            stored = Response.from_dict(att["response"])
        except (KeyError, ValueError) as exc:
            reasons.append(f"Freivalds: malformed attestation body: {exc}")
            continue
        spec = challenge.matmuls[0]
        stored_result = stored.results[0]

        # Slice the C bytes for this round out of the concatenated output.
        c_bytes_len = len(base64.b64decode(stored_result.c_b64.encode("ascii")))
        c_chunk = output_bytes[cursor : cursor + c_bytes_len]
        cursor += c_bytes_len
        if len(c_chunk) != c_bytes_len:
            reasons.append(f"Freivalds: output bytes too short for matmul {spec.id}")
            continue

        rebuilt = Response(
            challenge_id=challenge.challenge_id,
            backend=stored.backend,
            results=(
                MatmulResult(
                    id=spec.id,
                    digest_a=stored_result.digest_a,
                    digest_b=stored_result.digest_b,
                    digest_c=freivalds_prng.matrix_digest(c_chunk),
                    c_b64=base64.b64encode(c_chunk).decode("ascii"),
                    wall_time_ms=0.0,
                ),
            ),
        )
        report = verify_response(
            challenge,
            rebuilt,
            backend,
            r_seed_source=lambda: 0xBEEF,
        )
        if not report.overall_passed:
            failing = [m for m in report.matmuls if not m.passed]
            why = failing[0].reason if failing else "unknown"
            reasons.append(f"Freivalds check failed for matmul {spec.id}: {why}")

    return VerdictResult(passed=not reasons, reasons=reasons)
