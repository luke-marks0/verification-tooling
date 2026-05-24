from __future__ import annotations

import unittest

from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError as PydanticValidationError

from modules.core.common.contracts import validate_with_schema
from modules.core.common.deterministic import canonical_json_bytes
from modules.attestation.proverdet.wire import (
    Artifact,
    ArtifactTarget,
    ErasureEvidence,
    ErasureSpec,
    Graph,
    PowStreamEntry,
    ProofOfWorkSpec,
    ReplayEvidence,
    ReplayOutput,
    ReplayRequest,
    Task,
    TaskTarget,
    TranscriptEntry,
    Transmission,
)


def _make_graph() -> Graph:
    return Graph(
        graph_version="v1-placeholder",
        run_id="demo-001",
        produced_at="2026-05-04T12:00:00Z",
        tasks=[Task(task_id="t-0", pod_id="pod-a", operation="inference", claimed_flops=1024)],
        artifacts=[Artifact(artifact_id="a-0", commitment="sha256:" + "0" * 64, size_bytes=128)],
        transmissions=[
            Transmission(
                transmission_id="x-0",
                sender_pod_id="pod-a",
                receiver_pod_id="pod-b",
                artifact_id="a-0",
                tap_signature="deadbeef" * 16,
            )
        ],
    )


def _make_replay_request() -> ReplayRequest:
    return ReplayRequest(
        replay_id="r-1",
        pod_id="pod-a",
        target=TaskTarget(kind="task", task_id="t-0"),
        erasure=ErasureSpec(challenge_seed="deadbeef", deadline_ms=1000, rounds=4),
        proof_of_work=ProofOfWorkSpec(matmul_dim=64, dtype="bf16", rounds=3, report_every_ms=100),
        auxiliary=[],
    )


def _make_replay_evidence() -> ReplayEvidence:
    return ReplayEvidence(
        replay_id="r-1",
        produced_at="2026-05-04T12:00:00Z",
        output=ReplayOutput(
            commitment="sha256:" + "0" * 64,
            bytes_b64="c3R1Yi1vdXRwdXQ=",
        ),
        erasure_evidence=ErasureEvidence(rounds=4, passed=4, log_path="erasure.jsonl"),
        pow_stream=[
            PowStreamEntry(
                t_ms=100,
                freivalds_attestation_id="att-0",
                matmul_dim=64,
                rounds=3,
                dtype="bf16",
            )
        ],
    )


def _make_transcript_entry() -> TranscriptEntry:
    return TranscriptEntry(
        seq=1,
        direction="sent",
        endpoint="/graph",
        timestamp="2026-05-04T12:00:00Z",
        payload_digest="sha256:" + "0" * 64,
    )


class TestGraphModel(unittest.TestCase):
    def test_graph_round_trip_via_canonical_bytes(self) -> None:
        g = _make_graph()
        once = canonical_json_bytes(g.model_dump(exclude_none=True))
        again = Graph.model_validate_json(once)
        self.assertEqual(again, g)

    def test_graph_validates_against_schema(self) -> None:
        g = _make_graph()
        validate_with_schema("prover_graph.v1.schema.json", g.model_dump(exclude_none=True))

    def test_graph_rejects_bad_commitment(self) -> None:
        with self.assertRaises(PydanticValidationError):
            Artifact(artifact_id="a", commitment="not-prefixed", size_bytes=1)

    def test_graph_rejects_negative_claimed_flops(self) -> None:
        with self.assertRaises(PydanticValidationError):
            Task(task_id="t", pod_id="p", operation="inf", claimed_flops=-1)


class TestReplayRequestModel(unittest.TestCase):
    def test_request_round_trip(self) -> None:
        req = _make_replay_request()
        once = canonical_json_bytes(req.model_dump(exclude_none=True))
        again = ReplayRequest.model_validate_json(once)
        self.assertEqual(again, req)

    def test_request_validates_against_schema(self) -> None:
        req = _make_replay_request()
        validate_with_schema("replay_request.v1.schema.json", req.model_dump(exclude_none=True))

    def test_artifact_target_round_trip(self) -> None:
        req = _make_replay_request().model_copy(
            update={"target": ArtifactTarget(kind="artifact", artifact_id="a-0")}
        )
        once = canonical_json_bytes(req.model_dump(exclude_none=True))
        again = ReplayRequest.model_validate_json(once)
        self.assertEqual(again, req)
        validate_with_schema("replay_request.v1.schema.json", req.model_dump(exclude_none=True))

    def test_request_rejects_bad_dtype(self) -> None:
        with self.assertRaises(PydanticValidationError):
            ProofOfWorkSpec(matmul_dim=64, dtype="fp64", rounds=1, report_every_ms=100)


class TestReplayEvidenceModel(unittest.TestCase):
    def test_evidence_round_trip(self) -> None:
        ev = _make_replay_evidence()
        once = canonical_json_bytes(ev.model_dump(exclude_none=True))
        again = ReplayEvidence.model_validate_json(once)
        self.assertEqual(again, ev)

    def test_evidence_validates_against_schema(self) -> None:
        ev = _make_replay_evidence()
        validate_with_schema("replay_evidence.v1.schema.json", ev.model_dump(exclude_none=True))

    def test_evidence_rejects_bad_commitment(self) -> None:
        with self.assertRaises(PydanticValidationError):
            ReplayOutput(commitment="not-a-digest", bytes_b64="")


class TestTranscriptEntryModel(unittest.TestCase):
    def test_entry_round_trip(self) -> None:
        e = _make_transcript_entry()
        once = canonical_json_bytes(e.model_dump(exclude_none=True))
        again = TranscriptEntry.model_validate_json(once)
        self.assertEqual(again, e)

    def test_entry_validates_against_schema(self) -> None:
        e = _make_transcript_entry()
        validate_with_schema(
            "verifier_transcript_entry.v1.schema.json", e.model_dump(exclude_none=True)
        )

    def test_entry_rejects_unknown_direction(self) -> None:
        with self.assertRaises(PydanticValidationError):
            TranscriptEntry(
                seq=1,
                direction="internal",  # type: ignore[arg-type]
                endpoint="/graph",
                timestamp="2026-05-04T12:00:00Z",
                payload_digest="sha256:" + "0" * 64,
            )


# -- Property tests for canonical-JSON round-trip --


sha256_strategy = st.from_regex(r"^sha256:[0-9a-f]{64}$", fullmatch=True)
hex_strategy = st.from_regex(r"^[0-9a-f]+$", fullmatch=True).filter(lambda s: len(s) > 0)
iso_ts_strategy = st.just("2026-05-04T12:00:00Z")


@st.composite
def _graph_strategy(draw: st.DrawFn) -> Graph:
    run_id = draw(
        st.text(
            min_size=1, max_size=32, alphabet=st.characters(min_codepoint=33, max_codepoint=126)
        )
    )
    return Graph(
        graph_version="v1-placeholder",
        run_id=run_id,
        produced_at=draw(iso_ts_strategy),
        tasks=[],
        artifacts=[],
        transmissions=[],
    )


@st.composite
def _replay_request_strategy(draw: st.DrawFn) -> ReplayRequest:
    return ReplayRequest(
        replay_id=draw(st.text(min_size=1, max_size=16)),
        pod_id=draw(st.text(min_size=1, max_size=16)),
        target=TaskTarget(kind="task", task_id=draw(st.text(min_size=1, max_size=16))),
        erasure=ErasureSpec(
            challenge_seed=draw(hex_strategy),
            deadline_ms=draw(st.integers(min_value=1, max_value=10_000)),
            rounds=draw(st.integers(min_value=1, max_value=32)),
        ),
        proof_of_work=ProofOfWorkSpec(
            matmul_dim=draw(st.integers(min_value=1, max_value=512)),
            dtype=draw(st.sampled_from(["bf16", "fp16", "int8"])),
            rounds=draw(st.integers(min_value=1, max_value=32)),
            report_every_ms=draw(st.integers(min_value=1, max_value=10_000)),
        ),
        auxiliary=[],
    )


@st.composite
def _replay_evidence_strategy(draw: st.DrawFn) -> ReplayEvidence:
    return ReplayEvidence(
        replay_id=draw(st.text(min_size=1, max_size=16)),
        produced_at=draw(iso_ts_strategy),
        output=ReplayOutput(
            commitment=draw(sha256_strategy),
            bytes_b64=draw(st.text(max_size=64)),
        ),
        erasure_evidence=ErasureEvidence(
            rounds=draw(st.integers(min_value=0, max_value=32)),
            passed=draw(st.integers(min_value=0, max_value=32)),
            log_path=draw(st.text(min_size=1, max_size=32)),
        ),
        pow_stream=[],
    )


@st.composite
def _transcript_entry_strategy(draw: st.DrawFn) -> TranscriptEntry:
    return TranscriptEntry(
        seq=draw(st.integers(min_value=0, max_value=10_000)),
        direction=draw(st.sampled_from(["sent", "received"])),
        endpoint=draw(st.sampled_from(["/graph", "/replay", "/traffic"])),
        timestamp=draw(iso_ts_strategy),
        payload_digest=draw(sha256_strategy),
    )


class TestPropertyRoundTrip(unittest.TestCase):
    @given(_graph_strategy())
    def test_graph_canonical_roundtrip_is_fixed_point(self, g: Graph) -> None:
        once = canonical_json_bytes(g.model_dump(exclude_none=True))
        twice = canonical_json_bytes(Graph.model_validate_json(once).model_dump(exclude_none=True))
        self.assertEqual(once, twice)

    @given(_replay_request_strategy())
    def test_replay_request_canonical_roundtrip_is_fixed_point(self, r: ReplayRequest) -> None:
        once = canonical_json_bytes(r.model_dump(exclude_none=True))
        twice = canonical_json_bytes(
            ReplayRequest.model_validate_json(once).model_dump(exclude_none=True)
        )
        self.assertEqual(once, twice)

    @given(_replay_evidence_strategy())
    def test_replay_evidence_canonical_roundtrip_is_fixed_point(self, e: ReplayEvidence) -> None:
        once = canonical_json_bytes(e.model_dump(exclude_none=True))
        twice = canonical_json_bytes(
            ReplayEvidence.model_validate_json(once).model_dump(exclude_none=True)
        )
        self.assertEqual(once, twice)

    @given(_transcript_entry_strategy())
    def test_transcript_canonical_roundtrip_is_fixed_point(self, e: TranscriptEntry) -> None:
        once = canonical_json_bytes(e.model_dump(exclude_none=True))
        twice = canonical_json_bytes(
            TranscriptEntry.model_validate_json(once).model_dump(exclude_none=True)
        )
        self.assertEqual(once, twice)


class TestToCanonicalHelpers(unittest.TestCase):
    def test_graph_to_canonical_matches_schema(self) -> None:
        g = _make_graph()
        text = g.to_canonical()
        # Must be canonical (sorted keys, no spaces, trailing newline).
        self.assertTrue(text.endswith("\n"))
        # Re-parse through Pydantic; equal objects.
        again = Graph.model_validate_json(text)
        self.assertEqual(again, g)

    def test_replay_request_to_canonical(self) -> None:
        r = _make_replay_request()
        text = r.to_canonical()
        self.assertTrue(text.endswith("\n"))
        again = ReplayRequest.model_validate_json(text)
        self.assertEqual(again, r)

    def test_replay_evidence_to_canonical(self) -> None:
        e = _make_replay_evidence()
        text = e.to_canonical()
        self.assertTrue(text.endswith("\n"))
        again = ReplayEvidence.model_validate_json(text)
        self.assertEqual(again, e)

    def test_transcript_entry_to_canonical(self) -> None:
        e = _make_transcript_entry()
        text = e.to_canonical()
        self.assertTrue(text.endswith("\n"))
        again = TranscriptEntry.model_validate_json(text)
        self.assertEqual(again, e)


if __name__ == "__main__":
    unittest.main()
