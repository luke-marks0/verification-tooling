from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from modules.attestation.freivalds.backends.stdlib import StdlibBackend
from modules.attestation.proverdet.attestation_store import AttestationStore
from modules.attestation.proverdet.replay import produce_evidence
from modules.attestation.proverdet.replay_verify import VerdictResult, verify_evidence
from modules.attestation.proverdet.wire import (
    ErasureSpec,
    ProofOfWorkSpec,
    ReplayEvidence,
    ReplayRequest,
    TaskTarget,
)


def _make_request(replay_id: str = "v-1", *, rounds: int = 2) -> ReplayRequest:
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


class TestVerifyEvidence(unittest.TestCase):
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

    def _verify(self, req: ReplayRequest, ev: ReplayEvidence) -> VerdictResult:
        return verify_evidence(req, ev, fetch_attestation=self.store.get, backend=self.backend)

    def test_honest_evidence_passes(self) -> None:
        req = _make_request(rounds=3)
        ev = self._produce(req)
        verdict = self._verify(req, ev)
        self.assertTrue(verdict.passed, verdict.reasons)
        self.assertEqual(verdict.reasons, [])

    def test_tampered_output_bytes_fails(self) -> None:
        req = _make_request(rounds=2)
        ev = self._produce(req)
        # Decode → flip → re-encode.
        raw = bytearray(base64.b64decode(ev.output.bytes_b64))
        raw[0] = (raw[0] + 1) & 0xFF
        tampered_b64 = base64.b64encode(bytes(raw)).decode("ascii")
        # Pydantic models are frozen, so build a fresh one.
        tampered = ev.model_copy(
            update={"output": ev.output.model_copy(update={"bytes_b64": tampered_b64})}
        )
        verdict = self._verify(req, tampered)
        self.assertFalse(verdict.passed)
        self.assertTrue(
            any("Freivalds" in r for r in verdict.reasons),
            verdict.reasons,
        )

    def test_missing_pow_stream_fails_cadence(self) -> None:
        req = _make_request(rounds=3)
        ev = self._produce(req)
        empty = ev.model_copy(update={"pow_stream": []})
        verdict = self._verify(req, empty)
        self.assertFalse(verdict.passed)
        self.assertTrue(
            any("cadence" in r for r in verdict.reasons),
            verdict.reasons,
        )

    def test_unknown_attestation_id_fails(self) -> None:
        req = _make_request(rounds=1)
        ev = self._produce(req)
        # Use a fetcher that always returns None.
        verdict = verify_evidence(req, ev, fetch_attestation=lambda _id: None, backend=self.backend)
        self.assertFalse(verdict.passed)
        self.assertTrue(any("attestation" in r for r in verdict.reasons), verdict.reasons)

    def test_fewer_erasure_rounds_passed_fails(self) -> None:
        req = _make_request(rounds=1)
        ev = self._produce(req)
        downgraded = ev.model_copy(
            update={
                "erasure_evidence": ev.erasure_evidence.model_copy(
                    update={"passed": ev.erasure_evidence.passed - 1}
                )
            }
        )
        verdict = self._verify(req, downgraded)
        self.assertFalse(verdict.passed)
        self.assertTrue(any("erasure" in r.lower() for r in verdict.reasons), verdict.reasons)


if __name__ == "__main__":
    unittest.main()
