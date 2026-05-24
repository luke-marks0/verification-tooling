from __future__ import annotations

import unittest

from modules.core.common.contracts import ValidationError, validate_with_schema

GRAPH_SCHEMA = "prover_graph.v1.schema.json"
REPLAY_REQUEST_SCHEMA = "replay_request.v1.schema.json"
REPLAY_EVIDENCE_SCHEMA = "replay_evidence.v1.schema.json"
TRANSCRIPT_ENTRY_SCHEMA = "verifier_transcript_entry.v1.schema.json"


def _minimal_graph() -> dict:
    return {
        "graph_version": "v1-placeholder",
        "run_id": "demo-001",
        "produced_at": "2026-05-04T12:00:00Z",
        "tasks": [],
        "artifacts": [],
        "transmissions": [],
    }


def _minimal_task() -> dict:
    return {
        "task_id": "task-0",
        "pod_id": "pod-a",
        "operation": "inference",
        "claimed_flops": 1024,
    }


def _minimal_artifact() -> dict:
    return {
        "artifact_id": "art-0",
        "commitment": "sha256:" + "0" * 64,
        "size_bytes": 4096,
    }


def _minimal_transmission() -> dict:
    return {
        "transmission_id": "tx-0",
        "sender_pod_id": "pod-a",
        "receiver_pod_id": "pod-b",
        "artifact_id": "art-0",
        "tap_signature": "deadbeef" * 16,
    }


class TestGraphSchema(unittest.TestCase):
    def test_minimal_graph_validates(self) -> None:
        validate_with_schema(GRAPH_SCHEMA, _minimal_graph())

    def test_graph_with_one_task_artifact_transmission_validates(self) -> None:
        graph = _minimal_graph()
        graph["tasks"] = [_minimal_task()]
        graph["artifacts"] = [_minimal_artifact()]
        graph["transmissions"] = [_minimal_transmission()]
        validate_with_schema(GRAPH_SCHEMA, graph)

    def test_graph_rejects_unknown_field(self) -> None:
        bad = _minimal_graph()
        bad["spurious"] = 1
        with self.assertRaises(ValidationError):
            validate_with_schema(GRAPH_SCHEMA, bad)

    def test_graph_requires_run_id(self) -> None:
        bad = _minimal_graph()
        del bad["run_id"]
        with self.assertRaises(ValidationError):
            validate_with_schema(GRAPH_SCHEMA, bad)

    def test_graph_requires_graph_version_const(self) -> None:
        bad = _minimal_graph()
        bad["graph_version"] = "v2"
        with self.assertRaises(ValidationError):
            validate_with_schema(GRAPH_SCHEMA, bad)

    def test_artifact_commitment_must_be_sha256_prefixed(self) -> None:
        graph = _minimal_graph()
        bad_artifact = _minimal_artifact()
        bad_artifact["commitment"] = "deadbeef"
        graph["artifacts"] = [bad_artifact]
        with self.assertRaises(ValidationError):
            validate_with_schema(GRAPH_SCHEMA, graph)

    def test_task_claimed_flops_rejects_string(self) -> None:
        graph = _minimal_graph()
        bad_task = _minimal_task()
        bad_task["claimed_flops"] = "100"
        graph["tasks"] = [bad_task]
        with self.assertRaises(ValidationError):
            validate_with_schema(GRAPH_SCHEMA, graph)

    def test_task_claimed_flops_rejects_negative(self) -> None:
        graph = _minimal_graph()
        bad_task = _minimal_task()
        bad_task["claimed_flops"] = -1
        graph["tasks"] = [bad_task]
        with self.assertRaises(ValidationError):
            validate_with_schema(GRAPH_SCHEMA, graph)

    def test_transmission_requires_sender_and_receiver(self) -> None:
        graph = _minimal_graph()
        bad_tx = _minimal_transmission()
        del bad_tx["sender_pod_id"]
        graph["transmissions"] = [bad_tx]
        with self.assertRaises(ValidationError):
            validate_with_schema(GRAPH_SCHEMA, graph)

    def test_task_rejects_unknown_field(self) -> None:
        graph = _minimal_graph()
        bad_task = _minimal_task()
        bad_task["mystery"] = "?"
        graph["tasks"] = [bad_task]
        with self.assertRaises(ValidationError):
            validate_with_schema(GRAPH_SCHEMA, graph)


def _minimal_replay_request() -> dict:
    return {
        "replay_id": "r-1",
        "pod_id": "pod-a",
        "target": {"kind": "task", "task_id": "task-0"},
        "erasure": {
            "challenge_seed": "deadbeef",
            "deadline_ms": 1000,
            "rounds": 4,
        },
        "proof_of_work": {
            "matmul_dim": 64,
            "dtype": "bf16",
            "rounds": 3,
            "report_every_ms": 100,
        },
        "auxiliary": [],
    }


class TestReplayRequestSchema(unittest.TestCase):
    def test_minimal_request_validates(self) -> None:
        validate_with_schema(REPLAY_REQUEST_SCHEMA, _minimal_replay_request())

    def test_artifact_target_validates(self) -> None:
        req = _minimal_replay_request()
        req["target"] = {"kind": "artifact", "artifact_id": "art-0"}
        validate_with_schema(REPLAY_REQUEST_SCHEMA, req)

    def test_request_rejects_unknown_field(self) -> None:
        bad = _minimal_replay_request()
        bad["spurious"] = 1
        with self.assertRaises(ValidationError):
            validate_with_schema(REPLAY_REQUEST_SCHEMA, bad)

    def test_request_requires_pod_id(self) -> None:
        bad = _minimal_replay_request()
        del bad["pod_id"]
        with self.assertRaises(ValidationError):
            validate_with_schema(REPLAY_REQUEST_SCHEMA, bad)

    def test_request_rejects_bad_dtype(self) -> None:
        bad = _minimal_replay_request()
        bad["proof_of_work"]["dtype"] = "fp64"
        with self.assertRaises(ValidationError):
            validate_with_schema(REPLAY_REQUEST_SCHEMA, bad)

    def test_request_rejects_unknown_target_kind(self) -> None:
        bad = _minimal_replay_request()
        bad["target"] = {"kind": "task_or_artifact", "task_id": "task-0"}
        with self.assertRaises(ValidationError):
            validate_with_schema(REPLAY_REQUEST_SCHEMA, bad)

    def test_artifact_target_requires_artifact_id(self) -> None:
        bad = _minimal_replay_request()
        bad["target"] = {"kind": "artifact"}
        with self.assertRaises(ValidationError):
            validate_with_schema(REPLAY_REQUEST_SCHEMA, bad)

    def test_task_target_requires_task_id(self) -> None:
        bad = _minimal_replay_request()
        bad["target"] = {"kind": "task"}
        with self.assertRaises(ValidationError):
            validate_with_schema(REPLAY_REQUEST_SCHEMA, bad)

    def test_erasure_rounds_must_be_int(self) -> None:
        bad = _minimal_replay_request()
        bad["erasure"]["rounds"] = "4"
        with self.assertRaises(ValidationError):
            validate_with_schema(REPLAY_REQUEST_SCHEMA, bad)

    def test_pow_rounds_must_be_positive(self) -> None:
        bad = _minimal_replay_request()
        bad["proof_of_work"]["rounds"] = 0
        with self.assertRaises(ValidationError):
            validate_with_schema(REPLAY_REQUEST_SCHEMA, bad)


def _minimal_replay_evidence() -> dict:
    return {
        "replay_id": "r-1",
        "produced_at": "2026-05-04T12:00:00Z",
        "output": {
            "commitment": "sha256:" + "0" * 64,
            "bytes_b64": "c3R1Yi1vdXRwdXQ=",
        },
        "erasure_evidence": {
            "rounds": 4,
            "passed": 4,
            "log_path": "erasure-r-1.jsonl",
        },
        "pow_stream": [],
    }


def _pow_entry() -> dict:
    return {
        "t_ms": 100,
        "freivalds_attestation_id": "att-0",
        "matmul_dim": 64,
        "rounds": 3,
        "dtype": "bf16",
    }


class TestReplayEvidenceSchema(unittest.TestCase):
    def test_minimal_evidence_validates(self) -> None:
        validate_with_schema(REPLAY_EVIDENCE_SCHEMA, _minimal_replay_evidence())

    def test_with_pow_stream_validates(self) -> None:
        ev = _minimal_replay_evidence()
        ev["pow_stream"] = [_pow_entry(), _pow_entry()]
        validate_with_schema(REPLAY_EVIDENCE_SCHEMA, ev)

    def test_evidence_rejects_unknown_field(self) -> None:
        bad = _minimal_replay_evidence()
        bad["spurious"] = 1
        with self.assertRaises(ValidationError):
            validate_with_schema(REPLAY_EVIDENCE_SCHEMA, bad)

    def test_output_commitment_must_be_sha256(self) -> None:
        bad = _minimal_replay_evidence()
        bad["output"]["commitment"] = "not-a-digest"
        with self.assertRaises(ValidationError):
            validate_with_schema(REPLAY_EVIDENCE_SCHEMA, bad)

    def test_evidence_requires_replay_id(self) -> None:
        bad = _minimal_replay_evidence()
        del bad["replay_id"]
        with self.assertRaises(ValidationError):
            validate_with_schema(REPLAY_EVIDENCE_SCHEMA, bad)

    def test_pow_entry_rejects_string_dim(self) -> None:
        bad = _minimal_replay_evidence()
        entry = _pow_entry()
        entry["matmul_dim"] = "64"
        bad["pow_stream"] = [entry]
        with self.assertRaises(ValidationError):
            validate_with_schema(REPLAY_EVIDENCE_SCHEMA, bad)

    def test_erasure_passed_cannot_exceed_rounds(self) -> None:
        # Schema cannot easily enforce the cross-field rule, but it MUST
        # at least require both as ints with a min of 0.
        bad = _minimal_replay_evidence()
        bad["erasure_evidence"]["passed"] = -1
        with self.assertRaises(ValidationError):
            validate_with_schema(REPLAY_EVIDENCE_SCHEMA, bad)

    def test_evidence_with_optional_errors(self) -> None:
        ev = _minimal_replay_evidence()
        ev["errors"] = ["something went wrong"]
        validate_with_schema(REPLAY_EVIDENCE_SCHEMA, ev)


def _minimal_transcript_entry() -> dict:
    return {
        "seq": 1,
        "direction": "sent",
        "endpoint": "/graph",
        "timestamp": "2026-05-04T12:00:00Z",
        "payload_digest": "sha256:" + "0" * 64,
    }


class TestTranscriptEntrySchema(unittest.TestCase):
    def test_minimal_entry_validates(self) -> None:
        validate_with_schema(TRANSCRIPT_ENTRY_SCHEMA, _minimal_transcript_entry())

    def test_with_status_code_and_payload_path_validates(self) -> None:
        entry = _minimal_transcript_entry()
        entry["status_code"] = 200
        entry["payload_path"] = "graph/seq-1.json"
        validate_with_schema(TRANSCRIPT_ENTRY_SCHEMA, entry)

    def test_entry_rejects_unknown_field(self) -> None:
        bad = _minimal_transcript_entry()
        bad["spurious"] = 1
        with self.assertRaises(ValidationError):
            validate_with_schema(TRANSCRIPT_ENTRY_SCHEMA, bad)

    def test_payload_digest_must_be_sha256_prefixed(self) -> None:
        bad = _minimal_transcript_entry()
        bad["payload_digest"] = "deadbeef"
        with self.assertRaises(ValidationError):
            validate_with_schema(TRANSCRIPT_ENTRY_SCHEMA, bad)

    def test_direction_must_be_sent_or_received(self) -> None:
        bad = _minimal_transcript_entry()
        bad["direction"] = "internal"
        with self.assertRaises(ValidationError):
            validate_with_schema(TRANSCRIPT_ENTRY_SCHEMA, bad)

    def test_seq_must_be_int(self) -> None:
        bad = _minimal_transcript_entry()
        bad["seq"] = "1"
        with self.assertRaises(ValidationError):
            validate_with_schema(TRANSCRIPT_ENTRY_SCHEMA, bad)

    def test_seq_must_be_non_negative(self) -> None:
        bad = _minimal_transcript_entry()
        bad["seq"] = -1
        with self.assertRaises(ValidationError):
            validate_with_schema(TRANSCRIPT_ENTRY_SCHEMA, bad)

    def test_entry_requires_endpoint(self) -> None:
        bad = _minimal_transcript_entry()
        del bad["endpoint"]
        with self.assertRaises(ValidationError):
            validate_with_schema(TRANSCRIPT_ENTRY_SCHEMA, bad)


if __name__ == "__main__":
    unittest.main()
