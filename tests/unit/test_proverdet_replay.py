from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from modules.core.common.contracts import validate_with_schema
from modules.attestation.freivalds import Response, verify_response
from modules.attestation.freivalds.backends.stdlib import StdlibBackend
from modules.attestation.proverdet.attestation_store import AttestationStore
from modules.attestation.proverdet.replay import produce_evidence
from modules.attestation.proverdet.wire import (
    ErasureSpec,
    ProofOfWorkSpec,
    ReplayEvidence,
    ReplayRequest,
    TaskTarget,
)


def _make_request(replay_id: str = "r-1", *, rounds: int = 2) -> ReplayRequest:
    return ReplayRequest(
        replay_id=replay_id,
        pod_id="pod-a",
        target=TaskTarget(kind="task", task_id="t-0"),
        erasure=ErasureSpec(challenge_seed="deadbeef", deadline_ms=1000, rounds=4),
        proof_of_work=ProofOfWorkSpec(
            matmul_dim=8, dtype="int8", rounds=rounds, report_every_ms=100
        ),
        auxiliary=[],
    )


class TestProduceEvidence(unittest.TestCase):
    def setUp(self) -> None:
        self.store = AttestationStore()
        self.backend = StdlibBackend()
        self.tmp = tempfile.TemporaryDirectory()
        self.erasure_dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _produce(self, req: ReplayRequest) -> ReplayEvidence:
        return produce_evidence(
            req,
            freivalds_backend=self.backend,
            attestation_store=self.store,
            erasure_log_dir=self.erasure_dir,
        )

    def test_returns_replay_evidence(self) -> None:
        ev = self._produce(_make_request())
        self.assertIsInstance(ev, ReplayEvidence)

    def test_replay_id_preserved(self) -> None:
        ev = self._produce(_make_request("r-42"))
        self.assertEqual(ev.replay_id, "r-42")

    def test_evidence_validates_against_schema(self) -> None:
        ev = self._produce(_make_request())
        validate_with_schema("replay_evidence.v1.schema.json", ev.model_dump(exclude_none=True))

    def test_pow_stream_has_one_entry_per_round(self) -> None:
        req = _make_request(rounds=3)
        ev = self._produce(req)
        self.assertEqual(len(ev.pow_stream), 3)
        for entry in ev.pow_stream:
            self.assertEqual(entry.matmul_dim, req.proof_of_work.matmul_dim)
            self.assertEqual(entry.dtype, req.proof_of_work.dtype)
            self.assertEqual(entry.rounds, 1)

    def test_pow_stream_ids_are_stored(self) -> None:
        ev = self._produce(_make_request())
        for entry in ev.pow_stream:
            stored = self.store.get(entry.freivalds_attestation_id)
            self.assertIsNotNone(stored)

    def test_t_ms_is_monotonically_nondecreasing(self) -> None:
        ev = self._produce(_make_request(rounds=4))
        ts = [e.t_ms for e in ev.pow_stream]
        self.assertEqual(ts, sorted(ts))

    def test_attestation_id_unique_per_entry(self) -> None:
        ev = self._produce(_make_request(rounds=3))
        ids = [e.freivalds_attestation_id for e in ev.pow_stream]
        self.assertEqual(len(set(ids)), len(ids))

    def test_distinct_replay_ids_produce_distinct_attestation_ids(self) -> None:
        ev1 = self._produce(_make_request("r-1"))
        ev2 = self._produce(_make_request("r-2"))
        ids1 = {e.freivalds_attestation_id for e in ev1.pow_stream}
        ids2 = {e.freivalds_attestation_id for e in ev2.pow_stream}
        self.assertTrue(ids1.isdisjoint(ids2))

    def test_stored_attestation_passes_freivalds_verification(self) -> None:
        req = _make_request(rounds=2)
        ev = self._produce(req)
        for entry in ev.pow_stream:
            stored = self.store.get(entry.freivalds_attestation_id)
            self.assertIsNotNone(stored)
            assert stored is not None  # narrow for pyright
            challenge = _challenge_from_stored(stored)
            response = Response.from_dict(stored["response"])
            report = verify_response(
                challenge,
                response,
                self.backend,
                r_seed_source=lambda: 0xBEEF,
            )
            self.assertTrue(report.overall_passed, report)

    def test_tampered_attestation_fails_freivalds_verification(self) -> None:
        ev = self._produce(_make_request(rounds=1))
        entry = ev.pow_stream[0]
        stored = self.store.get(entry.freivalds_attestation_id)
        self.assertIsNotNone(stored)
        assert stored is not None
        # Flip a single byte of the response's c_b64.
        response = Response.from_dict(stored["response"])
        result = response.results[0]
        bad_bytes = bytearray(base64.b64decode(result.c_b64))
        bad_bytes[0] = (bad_bytes[0] + 1) & 0xFF
        bad_b64 = base64.b64encode(bytes(bad_bytes)).decode("ascii")
        bad_dict = stored["response"]
        bad_dict["results"][0]["c_b64"] = bad_b64
        bad_response = Response.from_dict(bad_dict)
        challenge = _challenge_from_stored(stored)
        report = verify_response(
            challenge,
            bad_response,
            self.backend,
            r_seed_source=lambda: 0xBEEF,
        )
        self.assertFalse(report.overall_passed)

    def test_output_commitment_is_sha256_prefixed(self) -> None:
        ev = self._produce(_make_request())
        self.assertTrue(ev.output.commitment.startswith("sha256:"))
        self.assertEqual(len(ev.output.commitment), len("sha256:") + 64)

    def test_erasure_evidence_log_path_points_to_real_file(self) -> None:
        ev = self._produce(_make_request())
        self.assertTrue(Path(ev.erasure_evidence.log_path).exists())
        # All rounds pass on the honest path.
        self.assertEqual(ev.erasure_evidence.rounds, ev.erasure_evidence.passed)


def _challenge_from_stored(stored: dict[str, object]) -> object:
    """Reconstruct a Challenge from the per-attestation stored body."""
    from modules.attestation.freivalds import Challenge

    challenge_dict = stored["challenge"]
    return Challenge.from_dict(challenge_dict)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
