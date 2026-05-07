"""Freivalds check: cheap verification of ``A @ B == C``.

Two regimes:

  * **bitwise** (integer dtypes) — exact equality of ``A(Br)`` and ``Cr``.
  * **tolerance** (float dtypes) — ``|A(Br) - Cr|_inf <= atol + rtol * |Cr|_inf``.

The verifier draws ``r`` from a fresh source the prover never observes,
which is what makes the protocol attestation-sound. The vector dtype for
``r`` is the input dtype (``dtype_b``); the accumulator and output dtype
for the intermediate vectors are picked from the matmul spec.

This module is backend-agnostic: it works against anything that exposes
the small protocol described in :mod:`pkg.freivalds.backends`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pkg.freivalds.spec import (
    ComparisonMode,
    INTEGER_DTYPES,
    MatmulSpec,
)


@dataclass(frozen=True)
class CheckOutcome:
    passed: bool
    reason: str
    max_abs_diff: float
    cr_inf_norm: float


def freivalds_check(
    backend: Any,
    spec: MatmulSpec,
    A: Any,
    B: Any,
    C: Any,
    r_seed: int,
) -> CheckOutcome:
    """Run one Freivalds check on ``(A, B, C)`` for ``spec``.

    ``r_seed`` is the verifier's local seed for ``r``; the prover never
    sees ``r_seed`` (or ``r``). Returns a :class:`CheckOutcome` with the
    verdict and observed numerics.
    """
    # Generate r locally on the verifier. Use dtype_b so B*r is well-typed.
    r = backend.random_vector(r_seed, spec.dtype_b, spec.N)

    # Intermediate vectors live in the accumulator dtype throughout — we
    # must not downcast Br to dtype_b (would wrap int8 and lose precision)
    # nor downcast ABr/Cr to dtype_c (would cause spurious mismatches when
    # the true ABr exceeds dtype_c's range but C has already been wrapped).
    Br = backend.matvec(B, r, spec.dtype_acc, spec.dtype_acc)
    ABr = backend.matvec(A, Br, spec.dtype_acc, spec.dtype_acc)
    Cr = backend.matvec(C, r, spec.dtype_acc, spec.dtype_acc)

    cr_inf = backend.vec_inf_norm(Cr)
    diff_inf = backend.vec_max_abs_diff(ABr, Cr)

    if spec.comparison is ComparisonMode.BITWISE:
        if spec.dtype_c not in INTEGER_DTYPES:
            return CheckOutcome(
                passed=False,
                reason=f"bitwise mode requires integer dtype_c, got {spec.dtype_c}",
                max_abs_diff=diff_inf,
                cr_inf_norm=cr_inf,
            )
        if backend.vec_exact_equal(ABr, Cr):
            return CheckOutcome(passed=True, reason="bitwise match", max_abs_diff=0.0, cr_inf_norm=cr_inf)
        return CheckOutcome(
            passed=False,
            reason=f"bitwise mismatch: max_abs_diff={diff_inf}",
            max_abs_diff=diff_inf,
            cr_inf_norm=cr_inf,
        )

    # Tolerance mode.
    if spec.tolerance is None:
        return CheckOutcome(
            passed=False,
            reason="tolerance mode but no Tolerance set",
            max_abs_diff=diff_inf,
            cr_inf_norm=cr_inf,
        )
    threshold = spec.tolerance.atol + spec.tolerance.rtol * cr_inf
    if diff_inf <= threshold:
        return CheckOutcome(
            passed=True,
            reason=f"tolerance match: diff={diff_inf} <= {threshold}",
            max_abs_diff=diff_inf,
            cr_inf_norm=cr_inf,
        )
    return CheckOutcome(
        passed=False,
        reason=f"tolerance exceeded: diff={diff_inf} > {threshold}",
        max_abs_diff=diff_inf,
        cr_inf_norm=cr_inf,
    )
