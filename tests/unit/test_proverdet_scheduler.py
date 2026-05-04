from __future__ import annotations

import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path

from pkg.proverdet.scheduler import VerifierScheduler
from pkg.proverdet.transcript import TranscriptLog


class _FakeClient:
    """In-process double for the verifier→prover HTTP client.

    Records every call and returns canned responses. Used to test the
    scheduler's call sequence without spinning up a real server.
    """

    def __init__(self) -> None:
        self.graph_calls = 0
        self.replay_calls: list[str] = []  # replay_ids requested

    def get_graph(self) -> tuple[int, dict]:
        self.graph_calls += 1
        return 200, {
            "graph_version": "v1-placeholder",
            "run_id": "fake",
            "produced_at": "2026-05-04T12:00:00Z",
            "tasks": [],
            "artifacts": [],
            "transmissions": [],
        }

    def post_replay(self, request: dict) -> Iterator[tuple[int, dict]]:
        self.replay_calls.append(request["replay_id"])
        # Mimic the prover's NDJSON wire shape: one pow chunk per round
        # in the request, then a final evidence chunk.
        rounds = int(request["proof_of_work"]["rounds"])
        for i in range(rounds):
            yield (
                200,
                {
                    "kind": "pow",
                    "t_ms": (i + 1) * 10,
                    "freivalds_attestation_id": f"att-fake-{request['replay_id']}-{i}",
                    "matmul_dim": int(request["proof_of_work"]["matmul_dim"]),
                    "rounds": 1,
                    "dtype": str(request["proof_of_work"]["dtype"]),
                },
            )
        yield (
            200,
            {
                "kind": "evidence",
                "replay_id": request["replay_id"],
                "produced_at": "2026-05-04T12:00:00Z",
                "output": {"commitment": "sha256:" + "0" * 64, "bytes_b64": "AA=="},
                "erasure_evidence": {
                    "rounds": request["erasure"]["rounds"],
                    "passed": request["erasure"]["rounds"],
                    "log_path": "e.jsonl",
                },
                "pow_stream": [],
            },
        )


class _FakeClock:
    """Monotonic clock + sleep that records calls and advances itself."""

    def __init__(self) -> None:
        self.now_s = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.now_s

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now_s += seconds


class TestVerifierScheduler(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.transcript = TranscriptLog(Path(self.tmp.name) / "transcript.jsonl")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_runs_at_least_one_graph_and_one_replay(self) -> None:
        client = _FakeClient()
        clock = _FakeClock()
        s = VerifierScheduler(
            client=client,
            transcript=self.transcript,
            seed=7,
            graph_period_ms=100,
            replay_period_ms=200,
            clock=clock,
        )
        s.run_for_ticks(8)
        self.assertGreaterEqual(client.graph_calls, 1)
        self.assertGreaterEqual(len(client.replay_calls), 1)

    def test_seed_makes_request_pattern_reproducible(self) -> None:
        def replay_ids_with_seed(seed: int) -> list[str]:
            client = _FakeClient()
            clock = _FakeClock()
            s = VerifierScheduler(
                client=client,
                transcript=TranscriptLog(Path(self.tmp.name) / f"t-{seed}.jsonl"),
                seed=seed,
                graph_period_ms=100,
                replay_period_ms=200,
                clock=clock,
            )
            s.run_for_ticks(20)
            return client.replay_calls

        a = replay_ids_with_seed(7)
        b = replay_ids_with_seed(7)
        self.assertEqual(a, b)

    def test_different_seeds_produce_different_patterns(self) -> None:
        # Soft assertion: with 20 ticks, two seeds should diverge.
        def replay_ids_with_seed(seed: int) -> list[str]:
            client = _FakeClient()
            clock = _FakeClock()
            s = VerifierScheduler(
                client=client,
                transcript=TranscriptLog(Path(self.tmp.name) / f"t-{seed}.jsonl"),
                seed=seed,
                graph_period_ms=100,
                replay_period_ms=200,
                clock=clock,
            )
            s.run_for_ticks(20)
            return client.replay_calls

        self.assertNotEqual(replay_ids_with_seed(7), replay_ids_with_seed(99))

    def test_emits_transcript_entries_for_sent_and_received(self) -> None:
        client = _FakeClient()
        clock = _FakeClock()
        s = VerifierScheduler(
            client=client,
            transcript=self.transcript,
            seed=7,
            graph_period_ms=100,
            replay_period_ms=200,
            clock=clock,
        )
        s.run_for_ticks(5)

        import json

        lines = [json.loads(line) for line in self.transcript.path.read_text().splitlines()]
        directions = {e["direction"] for e in lines}
        self.assertEqual(directions, {"sent", "received"})

    def test_replay_id_sequence_is_unique(self) -> None:
        client = _FakeClient()
        clock = _FakeClock()
        s = VerifierScheduler(
            client=client,
            transcript=self.transcript,
            seed=7,
            graph_period_ms=100,
            replay_period_ms=200,
            clock=clock,
        )
        s.run_for_ticks(20)
        self.assertEqual(len(client.replay_calls), len(set(client.replay_calls)))


if __name__ == "__main__":
    unittest.main()
