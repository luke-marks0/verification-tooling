"""Streaming/strided protocol tests.

Round-trips a multi-matmul Challenge with ``matmuls_per_response=M``
through ``execute_streaming_challenge`` and ``verify_streaming_response``,
plus the chain-hash construction itself.
"""
from __future__ import annotations

import unittest

from pkg.freivalds import (
    ChainHashChunk,
    Challenge,
    ComparisonMode,
    GENESIS_CHAIN_HASH,
    MatmulSpec,
    Response,
    Tolerance,
    execute_streaming_challenge,
    fold_chain_hash,
    verify_streaming_response,
)
from pkg.freivalds.backends.stdlib import StdlibBackend


def _make_challenge(n_matmuls: int, matmuls_per_response: int | None = None) -> Challenge:
    matmuls = tuple(
        MatmulSpec(
            id=f"m{i}",
            M=4, K=6, N=5,
            dtype_a="int8", dtype_b="int8", dtype_acc="int32", dtype_c="int32",
            seed_a=10 + i, seed_b=20 + i,
            comparison=ComparisonMode.BITWISE,
        )
        for i in range(n_matmuls)
    )
    return Challenge(
        challenge_id="chal-stream-001",
        matmuls=matmuls,
        matmuls_per_response=matmuls_per_response,
    )


class TestChainHashFold(unittest.TestCase):
    def test_genesis_distinct_from_matrix_digest(self) -> None:
        # The genesis tag must not collide with anything matrix_digest can
        # produce — different domain-separation prefix.
        from pkg.freivalds import prng
        for seed in range(8):
            self.assertNotEqual(GENESIS_CHAIN_HASH,
                                prng.matrix_digest(seed.to_bytes(8, "little")))

    def test_fold_is_deterministic(self) -> None:
        d = "sha256:" + "a" * 64
        self.assertEqual(fold_chain_hash(GENESIS_CHAIN_HASH, d),
                         fold_chain_hash(GENESIS_CHAIN_HASH, d))

    def test_fold_order_matters(self) -> None:
        d1 = "sha256:" + "a" * 64
        d2 = "sha256:" + "b" * 64
        h12 = fold_chain_hash(fold_chain_hash(GENESIS_CHAIN_HASH, d1), d2)
        h21 = fold_chain_hash(fold_chain_hash(GENESIS_CHAIN_HASH, d2), d1)
        self.assertNotEqual(h12, h21)

    def test_fold_perturbation_propagates(self) -> None:
        d1 = "sha256:" + "a" * 64
        d2_a = "sha256:" + "b" * 64
        d2_b = "sha256:" + "c" * 64
        h_a = fold_chain_hash(fold_chain_hash(GENESIS_CHAIN_HASH, d1), d2_a)
        h_b = fold_chain_hash(fold_chain_hash(GENESIS_CHAIN_HASH, d1), d2_b)
        self.assertNotEqual(h_a, h_b)


class TestStreamingHonest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = StdlibBackend()

    def test_round_trip_M_equals_K_minus_1(self) -> None:
        challenge = _make_challenge(n_matmuls=6, matmuls_per_response=2)
        response = execute_streaming_challenge(challenge, self.backend)
        self.assertEqual(len(response.chain_hashes), 3)
        self.assertEqual(response.results, ())
        report = verify_streaming_response(challenge, response, self.backend)
        self.assertTrue(report.overall_passed, msg=str(report.to_dict()))
        for v in report.matmuls:
            self.assertTrue(v.passed)

    def test_round_trip_M_equals_one(self) -> None:
        # Pathological refresh-rate-1 case: every matmul gets its own chunk.
        challenge = _make_challenge(n_matmuls=4, matmuls_per_response=1)
        response = execute_streaming_challenge(challenge, self.backend)
        self.assertEqual(len(response.chain_hashes), 4)
        report = verify_streaming_response(challenge, response, self.backend)
        self.assertTrue(report.overall_passed)

    def test_round_trip_M_equals_K(self) -> None:
        # Stride = total matmuls: streaming with one chunk = single-shot
        # under streaming framing.
        challenge = _make_challenge(n_matmuls=4, matmuls_per_response=4)
        response = execute_streaming_challenge(challenge, self.backend)
        self.assertEqual(len(response.chain_hashes), 1)
        report = verify_streaming_response(challenge, response, self.backend)
        self.assertTrue(report.overall_passed)

    def test_uneven_chunk_at_tail(self) -> None:
        # 5 matmuls / stride 2 → chunks of size [2, 2, 1].
        challenge = _make_challenge(n_matmuls=5, matmuls_per_response=2)
        response = execute_streaming_challenge(challenge, self.backend)
        self.assertEqual(len(response.chain_hashes), 3)
        self.assertEqual(len(response.chain_hashes[-1].matmul_ids), 1)
        report = verify_streaming_response(challenge, response, self.backend)
        self.assertTrue(report.overall_passed)


class TestStreamingDetection(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = StdlibBackend()
        self.challenge = _make_challenge(n_matmuls=4, matmuls_per_response=2)
        self.response = execute_streaming_challenge(self.challenge, self.backend)

    def test_tampered_chain_hash_caught(self) -> None:
        # Flip one chain hash; the verifier recomputes locally and rejects.
        bad_chunk = ChainHashChunk(
            chunk_index=0,
            matmul_ids=self.response.chain_hashes[0].matmul_ids,
            chain_hash="sha256:" + ("0" * 64),
            wall_time_ms=self.response.chain_hashes[0].wall_time_ms,
        )
        bad = Response(
            challenge_id=self.response.challenge_id,
            backend=self.response.backend,
            results=(),
            chain_hashes=(bad_chunk,) + self.response.chain_hashes[1:],
        )
        report = verify_streaming_response(self.challenge, bad, self.backend)
        self.assertFalse(report.overall_passed)
        # Both matmuls in the bad chunk fail; the second chunk's matmuls pass.
        passed = {v.id: v.passed for v in report.matmuls}
        self.assertFalse(passed["m0"])
        self.assertFalse(passed["m1"])
        self.assertTrue(passed["m2"])
        self.assertTrue(passed["m3"])

    def test_missing_chunk_caught(self) -> None:
        bad = Response(
            challenge_id=self.response.challenge_id,
            backend=self.response.backend,
            results=(),
            chain_hashes=(self.response.chain_hashes[0],),  # drop chunk 1
        )
        report = verify_streaming_response(self.challenge, bad, self.backend)
        self.assertFalse(report.overall_passed)


class TestStreamingWireFormat(unittest.TestCase):
    def test_challenge_round_trip_through_dict(self) -> None:
        ch = _make_challenge(n_matmuls=6, matmuls_per_response=3)
        d = ch.to_dict()
        self.assertEqual(d["matmuls_per_response"], 3)
        ch2 = Challenge.from_dict(d)
        self.assertEqual(ch2, ch)

    def test_challenge_without_M_omits_field(self) -> None:
        ch = _make_challenge(n_matmuls=2, matmuls_per_response=None)
        d = ch.to_dict()
        self.assertNotIn("matmuls_per_response", d)
        ch2 = Challenge.from_dict(d)
        self.assertIsNone(ch2.matmuls_per_response)

    def test_response_round_trip_through_dict(self) -> None:
        ch = _make_challenge(n_matmuls=6, matmuls_per_response=2)
        backend = StdlibBackend()
        resp = execute_streaming_challenge(ch, backend)
        d = resp.to_dict()
        self.assertEqual(len(d["chain_hashes"]), 3)
        resp2 = Response.from_dict(d)
        self.assertEqual(resp2.chain_hashes, resp.chain_hashes)

    def test_M_validation(self) -> None:
        with self.assertRaises(ValueError):
            _make_challenge(n_matmuls=4, matmuls_per_response=0)
        with self.assertRaises(ValueError):
            _make_challenge(n_matmuls=4, matmuls_per_response=5)


if __name__ == "__main__":
    unittest.main()
