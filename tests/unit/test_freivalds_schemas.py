"""Schema validation for freivalds challenge + attestation v1 schemas.

Mirrors the pattern used by ``test_flop_bound_schema.py`` etc.
"""
from __future__ import annotations

import unittest

from pkg.common.contracts import ValidationError, validate_with_schema
from pkg.freivalds import (
    Challenge,
    ComparisonMode,
    MatmulSpec,
    Tolerance,
    execute_challenge,
    verify_response,
)
from pkg.freivalds.backends.stdlib import StdlibBackend


def _challenge() -> Challenge:
    return Challenge(
        challenge_id="chal.schema.001",
        matmuls=(
            MatmulSpec(
                id="m1", M=4, K=4, N=4,
                dtype_a="int8", dtype_b="int8", dtype_acc="int32", dtype_c="int32",
                seed_a=1, seed_b=2,
                comparison=ComparisonMode.BITWISE,
            ),
            MatmulSpec(
                id="m2", M=3, K=3, N=3,
                dtype_a="fp64", dtype_b="fp64", dtype_acc="fp64", dtype_c="fp64",
                seed_a=3, seed_b=4,
                comparison=ComparisonMode.TOLERANCE,
                tolerance=Tolerance(atol=1e-8, rtol=1e-8),
            ),
        ),
    )


class TestChallengeSchema(unittest.TestCase):
    def test_challenge_validates(self) -> None:
        validate_with_schema("freivalds_challenge.v1.schema.json", _challenge().to_dict())

    def test_missing_challenge_version_rejected(self) -> None:
        d = _challenge().to_dict()
        del d["challenge_version"]
        with self.assertRaises(ValidationError):
            validate_with_schema("freivalds_challenge.v1.schema.json", d)

    def test_unknown_dtype_rejected(self) -> None:
        d = _challenge().to_dict()
        d["matmuls"][0]["dtype_a"] = "bogus"
        with self.assertRaises(ValidationError):
            validate_with_schema("freivalds_challenge.v1.schema.json", d)

    def test_extra_field_on_matmul_rejected(self) -> None:
        d = _challenge().to_dict()
        d["matmuls"][0]["extra"] = "no"
        with self.assertRaises(ValidationError):
            validate_with_schema("freivalds_challenge.v1.schema.json", d)


class TestAttestationSchema(unittest.TestCase):
    def test_honest_report_validates(self) -> None:
        backend = StdlibBackend()
        ch = _challenge()
        resp = execute_challenge(ch, backend)
        report = verify_response(ch, resp, backend, r_seed_source=lambda: 7)
        validate_with_schema("freivalds_attestation.v1.schema.json", report.to_dict())

    def test_missing_required_field_rejected(self) -> None:
        bad = {
            "attestation_version": "v1",
            "challenge_id": "chal.schema.001",
            "backend": "stdlib",
            "overall_passed": True,
            # missing matmuls + generated_at
        }
        with self.assertRaises(ValidationError):
            validate_with_schema("freivalds_attestation.v1.schema.json", bad)


if __name__ == "__main__":
    unittest.main()
