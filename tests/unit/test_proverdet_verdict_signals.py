from __future__ import annotations

import unittest

from pkg.proverdet.verdict import SignalResult, replay_correctness


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


if __name__ == "__main__":
    unittest.main()
