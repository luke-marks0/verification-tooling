"""End-to-end protocol tests: prover -> wire format -> verifier.

Round-trips a multi-matmul Challenge through ``execute_challenge`` and
``verify_response`` using the stdlib backend. Also covers tampering on
the wire (digest mismatch, wrong C bytes) and PRNG-drift detection.
"""
from __future__ import annotations

import base64
import unittest

from pkg.freivalds import (
    Challenge,
    ComparisonMode,
    MatmulSpec,
    Tolerance,
    execute_challenge,
    verify_response,
)
from pkg.freivalds.backends.stdlib import StdlibBackend
from pkg.freivalds.spec import MatmulResult, Response


def _make_challenge() -> Challenge:
    return Challenge(
        challenge_id="chal-test-001",
        matmuls=(
            MatmulSpec(
                id="int8-bitwise",
                M=4, K=6, N=5,
                dtype_a="int8", dtype_b="int8", dtype_acc="int32", dtype_c="int32",
                seed_a=10, seed_b=11,
                comparison=ComparisonMode.BITWISE,
            ),
            MatmulSpec(
                id="fp64-tolerance",
                M=4, K=6, N=5,
                dtype_a="fp64", dtype_b="fp64", dtype_acc="fp64", dtype_c="fp64",
                seed_a=20, seed_b=21,
                comparison=ComparisonMode.TOLERANCE,
                tolerance=Tolerance(atol=1e-9, rtol=1e-9),
            ),
        ),
    )


def _det_seed_source():
    counter = {"i": 0}
    def next_seed() -> int:
        counter["i"] += 1
        return counter["i"]
    return next_seed


class TestHonestRoundTrip(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = StdlibBackend()
        self.challenge = _make_challenge()

    def test_honest_response_passes(self) -> None:
        response = execute_challenge(self.challenge, self.backend)
        report = verify_response(
            self.challenge, response, self.backend, r_seed_source=_det_seed_source()
        )
        self.assertTrue(report.overall_passed, msg=str(report.to_dict()))
        for v in report.matmuls:
            self.assertTrue(v.passed, msg=v.reason)
            self.assertTrue(v.digest_a_match)
            self.assertTrue(v.digest_b_match)
            self.assertGreaterEqual(v.wall_time_ms, 0.0)

    def test_round_trip_through_dict(self) -> None:
        response = execute_challenge(self.challenge, self.backend)
        d = response.to_dict()
        # Wire round-trip
        response2 = Response.from_dict(d)
        report = verify_response(
            self.challenge, response2, self.backend, r_seed_source=_det_seed_source()
        )
        self.assertTrue(report.overall_passed)

    def test_challenge_round_trip_through_dict(self) -> None:
        d = self.challenge.to_dict()
        ch2 = Challenge.from_dict(d)
        self.assertEqual(ch2, self.challenge)


class TestTamperingOnTheWire(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = StdlibBackend()
        self.challenge = _make_challenge()
        self.response = execute_challenge(self.challenge, self.backend)

    def _replace_result(self, result_id: str, new_result: MatmulResult) -> Response:
        results = list(self.response.results)
        for i, r in enumerate(results):
            if r.id == result_id:
                results[i] = new_result
                break
        return Response(
            challenge_id=self.response.challenge_id,
            backend=self.response.backend,
            results=tuple(results),
        )

    def test_tampered_C_bytes_caught_by_freivalds(self) -> None:
        # Flip a byte deep inside C; recompute the digest_c so it matches
        # (so we test the Freivalds layer, not the digest layer).
        target = next(r for r in self.response.results if r.id == "int8-bitwise")
        c_bytes = bytearray(base64.b64decode(target.c_b64))
        c_bytes[0] ^= 0x01  # tamper one entry
        from pkg.freivalds import prng
        new_digest = prng.matrix_digest(bytes(c_bytes))
        tampered = MatmulResult(
            id=target.id,
            digest_a=target.digest_a,
            digest_b=target.digest_b,
            digest_c=new_digest,
            c_b64=base64.b64encode(bytes(c_bytes)).decode("ascii"),
            wall_time_ms=target.wall_time_ms,
            device=target.device, device_name=target.device_name,
        )
        bad = self._replace_result("int8-bitwise", tampered)
        report = verify_response(self.challenge, bad, self.backend, r_seed_source=_det_seed_source())
        self.assertFalse(report.overall_passed)
        v = next(v for v in report.matmuls if v.id == "int8-bitwise")
        self.assertFalse(v.passed)
        self.assertIn("bitwise mismatch", v.reason)

    def test_C_bytes_mismatching_declared_digest_caught(self) -> None:
        target = next(r for r in self.response.results if r.id == "int8-bitwise")
        c_bytes = bytearray(base64.b64decode(target.c_b64))
        c_bytes[0] ^= 0x01
        # Don't update digest_c -- now declared digest doesn't match bytes.
        tampered = MatmulResult(
            id=target.id,
            digest_a=target.digest_a,
            digest_b=target.digest_b,
            digest_c=target.digest_c,  # stale
            c_b64=base64.b64encode(bytes(c_bytes)).decode("ascii"),
            wall_time_ms=target.wall_time_ms,
            device=target.device, device_name=target.device_name,
        )
        bad = self._replace_result("int8-bitwise", tampered)
        report = verify_response(self.challenge, bad, self.backend, r_seed_source=_det_seed_source())
        self.assertFalse(report.overall_passed)
        v = next(v for v in report.matmuls if v.id == "int8-bitwise")
        self.assertFalse(v.passed)
        self.assertIn("digest_c", v.reason)

    def test_prng_drift_caught_via_digest_a(self) -> None:
        target = next(r for r in self.response.results if r.id == "int8-bitwise")
        # Pretend the prover used a different A (digest doesn't match what
        # the verifier reproduces locally). We just lie about digest_a.
        wrong_digest_a = "sha256:" + ("0" * 64)
        tampered = MatmulResult(
            id=target.id,
            digest_a=wrong_digest_a,
            digest_b=target.digest_b,
            digest_c=target.digest_c,
            c_b64=target.c_b64,
            wall_time_ms=target.wall_time_ms,
            device=target.device, device_name=target.device_name,
        )
        bad = self._replace_result("int8-bitwise", tampered)
        report = verify_response(self.challenge, bad, self.backend, r_seed_source=_det_seed_source())
        self.assertFalse(report.overall_passed)
        v = next(v for v in report.matmuls if v.id == "int8-bitwise")
        self.assertFalse(v.passed)
        self.assertFalse(v.digest_a_match)
        self.assertIn("prng drift", v.reason)

    def test_missing_result_caught(self) -> None:
        bad_results = tuple(r for r in self.response.results if r.id != "fp64-tolerance")
        bad = Response(
            challenge_id=self.response.challenge_id,
            backend=self.response.backend,
            results=bad_results,
        )
        report = verify_response(self.challenge, bad, self.backend, r_seed_source=_det_seed_source())
        self.assertFalse(report.overall_passed)
        v = next(v for v in report.matmuls if v.id == "fp64-tolerance")
        self.assertFalse(v.passed)
        self.assertIn("missing", v.reason)

    def test_wrong_challenge_id_short_circuits(self) -> None:
        bad = Response(
            challenge_id="not-the-challenge",
            backend=self.response.backend,
            results=self.response.results,
        )
        report = verify_response(self.challenge, bad, self.backend, r_seed_source=_det_seed_source())
        self.assertFalse(report.overall_passed)
        self.assertEqual(len(report.matmuls), 0)


class TestSpecValidation(unittest.TestCase):
    def test_unsupported_dtype_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MatmulSpec(
                id="bad", M=2, K=2, N=2,
                dtype_a="weird", dtype_b="int8", dtype_acc="int32", dtype_c="int32",
                seed_a=0, seed_b=0,
                comparison=ComparisonMode.BITWISE,
            )

    def test_bitwise_with_float_dtype_c_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MatmulSpec(
                id="bad", M=2, K=2, N=2,
                dtype_a="fp64", dtype_b="fp64", dtype_acc="fp64", dtype_c="fp64",
                seed_a=0, seed_b=0,
                comparison=ComparisonMode.BITWISE,
            )

    def test_tolerance_mode_requires_tolerance(self) -> None:
        with self.assertRaises(ValueError):
            MatmulSpec(
                id="bad", M=2, K=2, N=2,
                dtype_a="fp64", dtype_b="fp64", dtype_acc="fp64", dtype_c="fp64",
                seed_a=0, seed_b=0,
                comparison=ComparisonMode.TOLERANCE,
            )

    def test_duplicate_matmul_ids_rejected(self) -> None:
        d = {
            "challenge_version": "v1",
            "challenge_id": "x",
            "matmuls": [
                MatmulSpec(
                    id="dup", M=2, K=2, N=2,
                    dtype_a="int8", dtype_b="int8", dtype_acc="int32", dtype_c="int32",
                    seed_a=0, seed_b=0, comparison=ComparisonMode.BITWISE,
                ).to_dict(),
                MatmulSpec(
                    id="dup", M=3, K=3, N=3,
                    dtype_a="int8", dtype_b="int8", dtype_acc="int32", dtype_c="int32",
                    seed_a=1, seed_b=1, comparison=ComparisonMode.BITWISE,
                ).to_dict(),
            ],
        }
        with self.assertRaises(ValueError):
            Challenge.from_dict(d)


if __name__ == "__main__":
    unittest.main()
