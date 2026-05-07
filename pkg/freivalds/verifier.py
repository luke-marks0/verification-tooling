"""Verifier side: validate a :class:`Response` against a :class:`Challenge`.

For each matmul in the challenge the verifier:

  1. Reproduces ``A`` and ``B`` from ``(seed_a, seed_b)`` locally and compares
     digests against what the prover reported. A mismatch here means the
     prover's PRNG implementation differs from ours — surfaced as a separate
     verdict so it can't be confused with a Freivalds failure.
  2. Decodes ``C`` from the response.
  3. Runs Freivalds with a fresh ``r_seed`` drawn locally — the prover never
     observes it.

The verifier needs a backend that can interpret the dtypes used by the
challenge. For pure-stdlib unit tests this is :class:`StdlibBackend` and
the dtypes are restricted to ``{int8, int32, fp64}``.
"""
from __future__ import annotations

import base64
import secrets

from pkg.common.deterministic import utc_now_iso
from pkg.freivalds.check import freivalds_check
from pkg.freivalds.spec import (
    AttestationReport,
    Challenge,
    MatmulVerdict,
    Response,
)
from pkg.freivalds import prng


def verify_response(
    challenge: Challenge,
    response: Response,
    backend,
    r_seed_source=None,
) -> AttestationReport:
    """Validate ``response`` against ``challenge`` using ``backend``.

    ``r_seed_source`` is a zero-arg callable returning a fresh integer seed.
    Defaults to :func:`secrets.randbits` (64 bits) — cryptographically random
    so the prover cannot predict ``r``. Tests can pass a deterministic source
    for reproducibility.
    """
    if response.challenge_id != challenge.challenge_id:
        return AttestationReport(
            challenge_id=challenge.challenge_id,
            backend=response.backend,
            overall_passed=False,
            matmuls=tuple(),
            generated_at=utc_now_iso(),
        )

    if r_seed_source is None:
        r_seed_source = lambda: secrets.randbits(64)

    spec_by_id = {m.id: m for m in challenge.matmuls}
    result_by_id = {r.id: r for r in response.results}

    verdicts: list[MatmulVerdict] = []
    overall = True
    for mid, spec in spec_by_id.items():
        result = result_by_id.get(mid)
        if result is None:
            verdicts.append(MatmulVerdict(
                id=mid, passed=False, reason="missing result for matmul",
                max_abs_diff=0.0, cr_inf_norm=0.0, wall_time_ms=0.0,
                digest_a_match=False, digest_b_match=False,
            ))
            overall = False
            continue

        # 1) Reproduce A, B locally; compare digests.
        A, A_bytes = backend.gen_matrix(spec.seed_a, spec.dtype_a, spec.M, spec.K)
        B, B_bytes = backend.gen_matrix(spec.seed_b, spec.dtype_b, spec.K, spec.N)
        local_digest_a = prng.matrix_digest(A_bytes)
        local_digest_b = prng.matrix_digest(B_bytes)
        digest_a_match = local_digest_a == result.digest_a
        digest_b_match = local_digest_b == result.digest_b

        if not (digest_a_match and digest_b_match):
            verdicts.append(MatmulVerdict(
                id=mid, passed=False,
                reason="prng drift: A/B digest mismatch (prover and verifier disagree on inputs)",
                max_abs_diff=0.0, cr_inf_norm=0.0,
                wall_time_ms=result.wall_time_ms,
                digest_a_match=digest_a_match, digest_b_match=digest_b_match,
            ))
            overall = False
            continue

        # 2) Decode C; check declared digest matches the bytes we received.
        try:
            C_bytes = base64.b64decode(result.c_b64.encode("ascii"), validate=True)
        except Exception as exc:
            verdicts.append(MatmulVerdict(
                id=mid, passed=False, reason=f"c_b64 decode error: {exc}",
                max_abs_diff=0.0, cr_inf_norm=0.0,
                wall_time_ms=result.wall_time_ms,
                digest_a_match=True, digest_b_match=True,
            ))
            overall = False
            continue
        if prng.matrix_digest(C_bytes) != result.digest_c:
            verdicts.append(MatmulVerdict(
                id=mid, passed=False,
                reason="c_b64 bytes don't match declared digest_c",
                max_abs_diff=0.0, cr_inf_norm=0.0,
                wall_time_ms=result.wall_time_ms,
                digest_a_match=True, digest_b_match=True,
            ))
            overall = False
            continue

        try:
            C = backend.read_matrix_from_bytes(C_bytes, spec.dtype_c, spec.M, spec.N)
        except Exception as exc:
            verdicts.append(MatmulVerdict(
                id=mid, passed=False, reason=f"C bytes -> matrix error: {exc}",
                max_abs_diff=0.0, cr_inf_norm=0.0,
                wall_time_ms=result.wall_time_ms,
                digest_a_match=True, digest_b_match=True,
            ))
            overall = False
            continue

        # 3) Freivalds. r_seed comes from a source the prover never sees.
        r_seed = int(r_seed_source())
        outcome = freivalds_check(backend, spec, A, B, C, r_seed)

        verdicts.append(MatmulVerdict(
            id=mid,
            passed=outcome.passed,
            reason=outcome.reason,
            max_abs_diff=outcome.max_abs_diff,
            cr_inf_norm=outcome.cr_inf_norm,
            wall_time_ms=result.wall_time_ms,
            digest_a_match=True,
            digest_b_match=True,
        ))
        if not outcome.passed:
            overall = False

    return AttestationReport(
        challenge_id=challenge.challenge_id,
        backend=response.backend,
        overall_passed=overall,
        matmuls=tuple(verdicts),
        generated_at=utc_now_iso(),
    )
