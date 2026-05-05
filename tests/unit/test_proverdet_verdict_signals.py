from __future__ import annotations

import unittest

from pkg.proverdet.verdict import SignalResult, compute_budget, replay_correctness


def _entry(
    seq: int,
    *,
    direction: str,
    endpoint: str,
    status_code: int | None = None,
) -> dict[str, object]:
    e: dict[str, object] = {
        "seq": seq,
        "direction": direction,
        "endpoint": endpoint,
        "timestamp": "2026-05-04T00:00:00Z",
        "payload_digest": "sha256:" + "0" * 64,
    }
    if status_code is not None:
        e["status_code"] = status_code
    return e


class TestReplayCorrectness(unittest.TestCase):
    def test_all_pass_returns_passed(self) -> None:
        entries = [
            _entry(1, direction="sent", endpoint="/replay"),
            _entry(2, direction="received", endpoint="/replay/verdict/r-1", status_code=200),
            _entry(3, direction="sent", endpoint="/replay"),
            _entry(4, direction="received", endpoint="/replay/verdict/r-2", status_code=200),
        ]
        result = replay_correctness(entries)
        self.assertIsInstance(result, SignalResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.reasons, [])

    def test_one_fail_returns_failed_with_replay_id(self) -> None:
        entries = [
            _entry(1, direction="received", endpoint="/replay/verdict/r-1", status_code=200),
            _entry(2, direction="received", endpoint="/replay/verdict/r-bad", status_code=422),
            _entry(3, direction="received", endpoint="/replay/verdict/r-2", status_code=200),
        ]
        result = replay_correctness(entries)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.reasons), 1)
        self.assertIn("r-bad", result.reasons[0])

    def test_no_verdict_entries_returns_passed(self) -> None:
        # If no /replay was issued there's nothing to fail. Phase 8.3's
        # combiner is what flips this to "unknown" when needed.
        entries = [_entry(1, direction="sent", endpoint="/graph")]
        result = replay_correctness(entries)
        self.assertTrue(result.passed)


class TestComputeBudget(unittest.TestCase):
    def _summaries(
        self, *, claimed: int, observed: int, replay_id: str = "r-1"
    ) -> list[dict[str, object]]:
        return [
            {"kind": "graph", "claimed_flops_total": claimed, "task_count": 1},
            {
                "kind": "replay_evidence",
                "replay_id": replay_id,
                "observed_flops": observed,
                "rounds": 1,
                "matmul_dim": 8,
            },
        ]

    def test_observed_below_claimed_passes(self) -> None:
        s = self._summaries(claimed=1000, observed=500)
        result = compute_budget(s, tolerance=0.10)
        self.assertTrue(result.passed)
        self.assertEqual(result.reasons, [])

    def test_observed_well_above_claimed_fails(self) -> None:
        s = self._summaries(claimed=1000, observed=2000)
        result = compute_budget(s, tolerance=0.10)
        self.assertFalse(result.passed)
        # Reason should mention the gap.
        self.assertTrue(any("compute" in r.lower() or "flop" in r.lower() for r in result.reasons))

    def test_observed_equal_to_tolerance_passes(self) -> None:
        # Predicate uses `>`, so equality at the boundary passes.
        s = self._summaries(claimed=1000, observed=1100)  # 1.10 * 1000
        result = compute_budget(s, tolerance=0.10)
        self.assertTrue(result.passed, result.reasons)

    def test_observed_one_above_tolerance_fails(self) -> None:
        s = self._summaries(claimed=1000, observed=1101)
        result = compute_budget(s, tolerance=0.10)
        self.assertFalse(result.passed)

    def test_no_graph_or_evidence_passes(self) -> None:
        # Nothing to check; compute_budget returns passed=True. Phase 8.3
        # combiner can flip the final verdict to "unknown" on thin input.
        result = compute_budget([], tolerance=0.10)
        self.assertTrue(result.passed)

    def test_zero_claimed_flops_with_observed_fails(self) -> None:
        # If the graph claimed nothing but the prover did real work, that
        # IS the cheating signature (any positive observed beats 1.10 * 0).
        s = self._summaries(claimed=0, observed=1)
        result = compute_budget(s, tolerance=0.10)
        self.assertFalse(result.passed)


class TestBandwidthSignal(unittest.TestCase):
    """Pin the zero-tolerance contract: bytes are bytes, no slack envelope."""

    def test_default_tolerance_is_zero(self) -> None:
        # Any extra byte fires the signal at the default tolerance. The
        # principled check: the verifier observes ground-truth bytes; a
        # claim of N bytes admits exactly N, not N+ε.
        from pkg.proverdet.verdict import bandwidth_signal

        # Equality at the boundary passes (predicate uses `>`).
        self.assertTrue(bandwidth_signal(traffic_size=1000, claimed_artifact_bytes=1000).passed)
        # One byte over fails — no tolerance to absorb it.
        result = bandwidth_signal(traffic_size=1001, claimed_artifact_bytes=1000)
        self.assertFalse(result.passed)
        self.assertTrue(any("bandwidth" in r.lower() for r in result.reasons), result.reasons)

    def test_zero_claim_returns_passed(self) -> None:
        # No baseline → can't draw a conclusion → defer to other signals.
        from pkg.proverdet.verdict import bandwidth_signal

        self.assertTrue(bandwidth_signal(traffic_size=99999, claimed_artifact_bytes=0).passed)

    def test_caller_can_still_widen_tolerance_for_bring_up(self) -> None:
        # The kwarg is kept for early-bring-up callers; production uses 0.0.
        from pkg.proverdet.verdict import bandwidth_signal

        self.assertTrue(
            bandwidth_signal(traffic_size=1100, claimed_artifact_bytes=1000, tolerance=0.10).passed
        )
        self.assertFalse(
            bandwidth_signal(traffic_size=1101, claimed_artifact_bytes=1000, tolerance=0.10).passed
        )


if __name__ == "__main__":
    unittest.main()
