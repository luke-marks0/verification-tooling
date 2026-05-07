"""Tests for the Freivalds check itself.

We feed the check honest A, B, C and a tampered C; expect honest=pass,
tampered=fail. Run on int (bitwise) and fp64 (tolerance) so both regimes
are covered.
"""
from __future__ import annotations

import unittest

from pkg.freivalds.backends.stdlib import StdlibBackend
from pkg.freivalds.check import freivalds_check
from pkg.freivalds.spec import ComparisonMode, MatmulSpec, Tolerance


def _build_spec(
    *,
    M=8, K=8, N=8,
    dtype_a="int8", dtype_b="int8", dtype_acc="int32", dtype_c="int32",
    comparison=ComparisonMode.BITWISE,
    tolerance=None,
) -> MatmulSpec:
    return MatmulSpec(
        id="m0", M=M, K=K, N=N,
        dtype_a=dtype_a, dtype_b=dtype_b, dtype_acc=dtype_acc, dtype_c=dtype_c,
        seed_a=1, seed_b=2,
        comparison=comparison, tolerance=tolerance,
    )


class TestFreivaldsBitwise(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = StdlibBackend()
        self.spec = _build_spec()
        self.A, _ = self.backend.gen_matrix(self.spec.seed_a, self.spec.dtype_a, self.spec.M, self.spec.K)
        self.B, _ = self.backend.gen_matrix(self.spec.seed_b, self.spec.dtype_b, self.spec.K, self.spec.N)
        self.C = self.backend.matmul(
            self.A, self.B,
            self.spec.dtype_a, self.spec.dtype_b,
            self.spec.dtype_acc, self.spec.dtype_c,
        )

    def test_honest_passes(self) -> None:
        out = freivalds_check(self.backend, self.spec, self.A, self.B, self.C, r_seed=12345)
        self.assertTrue(out.passed, msg=out.reason)
        self.assertEqual(out.max_abs_diff, 0.0)

    def test_zeroed_C_fails(self) -> None:
        C_bad = self.backend.zeros_matrix(self.spec.M, self.spec.N, self.spec.dtype_c)
        out = freivalds_check(self.backend, self.spec, self.A, self.B, C_bad, r_seed=12345)
        self.assertFalse(out.passed)
        self.assertGreater(out.max_abs_diff, 0)

    def test_single_entry_corruption_fails(self) -> None:
        C_bad = [row[:] for row in self.C]
        C_bad[0][0] = (C_bad[0][0] + 1) & 0xFFFFFFFF
        # Wrap to int32
        if C_bad[0][0] >= (1 << 31):
            C_bad[0][0] -= 1 << 32
        out = freivalds_check(self.backend, self.spec, self.A, self.B, C_bad, r_seed=12345)
        self.assertFalse(out.passed)

    def test_independent_of_r_seed_for_honest(self) -> None:
        # Honest run should pass for any r_seed.
        for seed in (0, 1, 42, 999, 2**31 - 1):
            out = freivalds_check(self.backend, self.spec, self.A, self.B, self.C, r_seed=seed)
            self.assertTrue(out.passed, msg=f"seed={seed}: {out.reason}")


class TestFreivaldsTolerance(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = StdlibBackend()
        # K is small; fp64 honest run drift is well below default tolerance.
        self.spec = _build_spec(
            M=6, K=10, N=6,
            dtype_a="fp64", dtype_b="fp64", dtype_acc="fp64", dtype_c="fp64",
            comparison=ComparisonMode.TOLERANCE,
            tolerance=Tolerance(atol=1e-9, rtol=1e-9),
        )
        self.A, _ = self.backend.gen_matrix(self.spec.seed_a, self.spec.dtype_a, self.spec.M, self.spec.K)
        self.B, _ = self.backend.gen_matrix(self.spec.seed_b, self.spec.dtype_b, self.spec.K, self.spec.N)
        self.C = self.backend.matmul(
            self.A, self.B,
            self.spec.dtype_a, self.spec.dtype_b,
            self.spec.dtype_acc, self.spec.dtype_c,
        )

    def test_honest_passes_within_tolerance(self) -> None:
        out = freivalds_check(self.backend, self.spec, self.A, self.B, self.C, r_seed=12345)
        self.assertTrue(out.passed, msg=out.reason)

    def test_perturbed_C_outside_tolerance_fails(self) -> None:
        C_bad = [row[:] for row in self.C]
        C_bad[2][3] += 0.1  # well above 1e-9 tolerance
        out = freivalds_check(self.backend, self.spec, self.A, self.B, C_bad, r_seed=12345)
        self.assertFalse(out.passed)

    def test_zeros_C_fails(self) -> None:
        C_bad = self.backend.zeros_matrix(self.spec.M, self.spec.N, self.spec.dtype_c)
        out = freivalds_check(self.backend, self.spec, self.A, self.B, C_bad, r_seed=12345)
        self.assertFalse(out.passed)


if __name__ == "__main__":
    unittest.main()
