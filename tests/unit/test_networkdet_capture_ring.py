"""Unit tests for the non-perturbing capture ring."""
from __future__ import annotations

import unittest

from modules.network.networkdet.capture import CaptureRing


class TestCaptureRing(unittest.TestCase):

    def test_record_and_drain(self):
        ring = CaptureRing()
        ring.record(b"\x01\x02\x03")
        ring.record(b"\x04\x05\x06")
        frames = ring.drain()
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0], b"\x01\x02\x03")
        self.assertEqual(frames[1], b"\x04\x05\x06")

    def test_drain_clears_ring(self):
        ring = CaptureRing()
        ring.record(b"\x01")
        ring.drain()
        self.assertEqual(ring.frame_count, 0)

    def test_record_copies_bytes(self):
        """Recording must copy the frame, not retain a reference."""
        ring = CaptureRing()
        frame = bytearray(b"\x01\x02\x03")
        ring.record(frame)
        frame[0] = 0xFF  # Mutate original.
        frames = ring.drain()
        self.assertEqual(frames[0], b"\x01\x02\x03")  # Unchanged.

    def test_digest_determinism(self):
        """Same frames always produce the same digest."""
        ring1 = CaptureRing()
        ring2 = CaptureRing()
        for frame in [b"frame1", b"frame2", b"frame3"]:
            ring1.record(frame)
            ring2.record(frame)
        self.assertEqual(ring1.digest(), ring2.digest())

    def test_digest_changes_with_different_frames(self):
        ring1 = CaptureRing()
        ring2 = CaptureRing()
        ring1.record(b"A")
        ring2.record(b"B")
        self.assertNotEqual(ring1.digest(), ring2.digest())

    def test_digest_changes_with_order(self):
        ring1 = CaptureRing()
        ring2 = CaptureRing()
        ring1.record(b"A")
        ring1.record(b"B")
        ring2.record(b"B")
        ring2.record(b"A")
        self.assertNotEqual(ring1.digest(), ring2.digest())

    def test_digest_prefix(self):
        ring = CaptureRing()
        ring.record(b"\x00")
        self.assertTrue(ring.digest().startswith("sha256:"))

    def test_frames_as_hex(self):
        ring = CaptureRing()
        ring.record(b"\xDE\xAD")
        ring.record(b"\xBE\xEF")
        result = ring.frames_as_hex()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["frame_index"], 0)
        self.assertEqual(result[0]["frame_hex"], "dead")
        self.assertEqual(result[1]["frame_index"], 1)
        self.assertEqual(result[1]["frame_hex"], "beef")

    def test_empty_digest(self):
        """Empty ring still produces a valid SHA-256 digest."""
        ring = CaptureRing()
        digest = ring.digest()
        self.assertTrue(digest.startswith("sha256:"))
        self.assertEqual(len(digest), len("sha256:") + 64)


if __name__ == "__main__":
    unittest.main()
