"""Unit tests for the deterministic L2+L3+L4 frame builder."""
from __future__ import annotations

import struct
import unittest

from modules.network.networkdet.capture import CaptureRing
from modules.network.networkdet.config import NetStackConfig
from modules.network.networkdet.ethernet import ETHERNET_HEADER_LEN, ETHERTYPE_IPV4
from modules.network.networkdet.frame import DeterministicFrameBuilder
from modules.network.networkdet.ip import IPV4_HEADER_LEN


def _test_config() -> NetStackConfig:
    return NetStackConfig(
        mtu=1500,
        mss=100,
        tso=False,
        gso=False,
        checksum_offload=False,
        thread_affinity=(0,),
        tx_queues=1,
        rx_queues=1,
        queue_mapping_policy="fixed_core_queue",
        ring_tx=512,
        ring_rx=512,
        internal_batching_enabled=False,
        internal_batching_max_burst=1,
        security_mode="plaintext",
        egress_reproducibility=True,
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        src_mac="02:00:00:00:00:01",
        dst_mac="02:00:00:00:00:02",
        src_port=8000,
        dst_port=80,
    )


class TestDeterministicFrameBuilder(unittest.TestCase):

    def test_build_response_frames_small_payload(self):
        """A small payload fits in one frame."""
        capture = CaptureRing()
        builder = DeterministicFrameBuilder(
            _test_config(), run_id="run-1", conn_index=0, capture_ring=capture,
        )
        frames = builder.build_response_frames(b"Hello")
        self.assertEqual(len(frames), 1)
        self.assertEqual(capture.frame_count, 1)

    def test_build_response_frames_large_payload(self):
        """A payload larger than MSS produces multiple frames."""
        capture = CaptureRing()
        builder = DeterministicFrameBuilder(
            _test_config(), run_id="run-1", conn_index=0, capture_ring=capture,
        )
        # MSS=100, so 250 bytes -> 3 segments -> 3 frames.
        frames = builder.build_response_frames(b"X" * 250)
        self.assertEqual(len(frames), 3)
        self.assertEqual(capture.frame_count, 3)

    def test_frame_structure(self):
        """Verify a frame has correct Ethernet + IP + TCP structure."""
        capture = CaptureRing()
        builder = DeterministicFrameBuilder(
            _test_config(), run_id="run-1", conn_index=0, capture_ring=capture,
        )
        frames = builder.build_response_frames(b"test data")
        frame = frames[0]

        # Ethernet header: 14 bytes.
        self.assertGreater(len(frame), ETHERNET_HEADER_LEN + IPV4_HEADER_LEN + 20)

        # EtherType should be IPv4.
        ethertype = struct.unpack("!H", frame[12:14])[0]
        self.assertEqual(ethertype, ETHERTYPE_IPV4)

        # IP version should be 4.
        ip_start = ETHERNET_HEADER_LEN
        version = (frame[ip_start] >> 4) & 0xF
        self.assertEqual(version, 4)

        # IP protocol should be TCP (6).
        protocol = frame[ip_start + 9]
        self.assertEqual(protocol, 6)

    def test_determinism_across_builders(self):
        """Two builders with same config and run_id produce identical frames."""
        capture1 = CaptureRing()
        capture2 = CaptureRing()
        builder1 = DeterministicFrameBuilder(
            _test_config(), run_id="run-X", conn_index=0, capture_ring=capture1,
        )
        builder2 = DeterministicFrameBuilder(
            _test_config(), run_id="run-X", conn_index=0, capture_ring=capture2,
        )
        data = b"deterministic test payload"
        frames1 = builder1.build_response_frames(data)
        frames2 = builder2.build_response_frames(data)
        self.assertEqual(frames1, frames2)
        self.assertEqual(capture1.digest(), capture2.digest())

    def test_different_conn_index_different_frames(self):
        """Different connection indices produce different frames (different ISN)."""
        capture1 = CaptureRing()
        capture2 = CaptureRing()
        builder1 = DeterministicFrameBuilder(
            _test_config(), run_id="run-1", conn_index=0, capture_ring=capture1,
        )
        builder2 = DeterministicFrameBuilder(
            _test_config(), run_id="run-1", conn_index=1, capture_ring=capture2,
        )
        data = b"same data"
        frames1 = builder1.build_response_frames(data)
        frames2 = builder2.build_response_frames(data)
        self.assertNotEqual(frames1, frames2)

    def test_capture_ring_records_all_frames(self):
        """The capture ring receives every frame produced."""
        capture = CaptureRing()
        builder = DeterministicFrameBuilder(
            _test_config(), run_id="run-1", conn_index=0, capture_ring=capture,
        )
        builder.build_response_frames(b"A" * 250)  # 3 frames.
        builder.build_response_frames(b"B" * 50)   # 1 frame.
        self.assertEqual(capture.frame_count, 4)

    def test_ip_id_increments(self):
        """IP identification field increments per packet."""
        capture = CaptureRing()
        builder = DeterministicFrameBuilder(
            _test_config(), run_id="run-1", conn_index=0, capture_ring=capture,
        )
        frames = builder.build_response_frames(b"A" * 250)  # 3 frames.
        ip_start = ETHERNET_HEADER_LEN
        ids = []
        for frame in frames:
            ip_id = struct.unpack("!H", frame[ip_start + 4 : ip_start + 6])[0]
            ids.append(ip_id)
        # IDs should be consecutive.
        self.assertEqual(ids, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
