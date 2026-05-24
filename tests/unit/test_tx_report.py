"""Unit tests for TxReport dataclass."""
from __future__ import annotations

import dataclasses
import unittest

from modules.network.networkdet.tx_report import TxReport


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


class TestTxReport(unittest.TestCase):

    def test_match_when_digests_equal(self):
        report = TxReport(
            pre_enqueue_digest=DIGEST_A,
            tx_completion_digest=DIGEST_A,
            frames_submitted=10,
            frames_confirmed=10,
        )
        self.assertTrue(report.match)

    def test_no_match_when_digests_differ(self):
        report = TxReport(
            pre_enqueue_digest=DIGEST_A,
            tx_completion_digest=DIGEST_B,
            frames_submitted=10,
            frames_confirmed=10,
        )
        self.assertFalse(report.match)

    def test_frozen(self):
        report = TxReport(
            pre_enqueue_digest=DIGEST_A,
            tx_completion_digest=DIGEST_A,
            frames_submitted=10,
            frames_confirmed=10,
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            report.pre_enqueue_digest = DIGEST_B  # type: ignore[misc]

    def test_level_is_tx_completion(self):
        report = TxReport(
            pre_enqueue_digest=DIGEST_A,
            tx_completion_digest=DIGEST_A,
            frames_submitted=1,
            frames_confirmed=1,
        )
        self.assertEqual(report.level, "tx_completion")

    def test_loopback_match_all_three_digests(self):
        report = TxReport(
            pre_enqueue_digest=DIGEST_A,
            tx_completion_digest=DIGEST_A,
            frames_submitted=10,
            frames_confirmed=10,
            rx_loopback_digest=DIGEST_A,
            rx_loopback_count=10,
        )
        self.assertTrue(report.match)

    def test_loopback_mismatch_rx(self):
        report = TxReport(
            pre_enqueue_digest=DIGEST_A,
            tx_completion_digest=DIGEST_A,
            frames_submitted=10,
            frames_confirmed=10,
            rx_loopback_digest=DIGEST_B,
            rx_loopback_count=10,
        )
        self.assertFalse(report.match)

    def test_level_is_loopback_when_rx_present(self):
        report = TxReport(
            pre_enqueue_digest=DIGEST_A,
            tx_completion_digest=DIGEST_A,
            frames_submitted=1,
            frames_confirmed=1,
            rx_loopback_digest=DIGEST_A,
            rx_loopback_count=1,
        )
        self.assertEqual(report.level, "loopback")

    def test_level_is_tx_completion_when_rx_absent(self):
        report = TxReport(
            pre_enqueue_digest=DIGEST_A,
            tx_completion_digest=DIGEST_A,
            frames_submitted=1,
            frames_confirmed=1,
        )
        self.assertEqual(report.level, "tx_completion")


if __name__ == "__main__":
    unittest.main()
