"""Unit tests for deterministic TCP state machine and segmentation."""
from __future__ import annotations

import struct
import unittest

from modules.network.networkdet.tcp import (
    ACK,
    FIN,
    PSH,
    SYN,
    DeterministicTCPConnection,
    TCPState,
    deterministic_isn,
)


class TestDeterministicISN(unittest.TestCase):

    def test_determinism(self):
        """Same run_id + conn_index always produces the same ISN."""
        isn1 = deterministic_isn("run-42", 0)
        isn2 = deterministic_isn("run-42", 0)
        self.assertEqual(isn1, isn2)

    def test_different_conn_index(self):
        """Different connection indices produce different ISNs."""
        isn0 = deterministic_isn("run-42", 0)
        isn1 = deterministic_isn("run-42", 1)
        self.assertNotEqual(isn0, isn1)

    def test_different_run_id(self):
        """Different run IDs produce different ISNs."""
        isn_a = deterministic_isn("run-A", 0)
        isn_b = deterministic_isn("run-B", 0)
        self.assertNotEqual(isn_a, isn_b)

    def test_range(self):
        """ISN is a 32-bit unsigned integer."""
        isn = deterministic_isn("test", 0)
        self.assertTrue(0 <= isn <= 0xFFFFFFFF)


class TestDeterministicTCPConnection(unittest.TestCase):

    def _make_conn(self, isn: int = 1000) -> DeterministicTCPConnection:
        return DeterministicTCPConnection(
            src_port=8000,
            dst_port=80,
            isn=isn,
            mss=100,
            window=512,
            src_ip=bytes([10, 0, 0, 1]),
            dst_ip=bytes([10, 0, 0, 2]),
        )

    def test_initial_state(self):
        conn = self._make_conn()
        self.assertEqual(conn.state, TCPState.CLOSED)

    def test_syn_transitions_to_syn_sent(self):
        conn = self._make_conn(isn=5000)
        syn = conn.build_syn()
        self.assertEqual(conn.state, TCPState.SYN_SENT)
        # SYN consumes one seq number.
        self.assertEqual(conn.seq, 5001)
        # Segment should have SYN flag set.
        flags = syn[13]
        self.assertTrue(flags & SYN)

    def test_syn_includes_mss_option(self):
        conn = self._make_conn()
        syn = conn.build_syn()
        # With MSS option, header is 24 bytes.
        data_offset = (syn[12] >> 4) * 4
        self.assertEqual(data_offset, 24)
        # MSS option: kind=2, len=4, value=100.
        mss_option = syn[20:24]
        kind, length, mss_val = struct.unpack("!BBH", mss_option)
        self.assertEqual(kind, 2)
        self.assertEqual(length, 4)
        self.assertEqual(mss_val, 100)

    def test_ack_after_syn_transitions_to_established(self):
        conn = self._make_conn()
        conn.build_syn()
        conn.receive_syn_ack(peer_seq=2000)
        ack = conn.build_ack()
        self.assertEqual(conn.state, TCPState.ESTABLISHED)
        flags = ack[13]
        self.assertTrue(flags & ACK)

    def test_segment_data_single_chunk(self):
        conn = self._make_conn(isn=100)
        conn.build_syn()
        conn.receive_syn_ack(2000)
        conn.build_ack()
        data = b"Hello, World!"  # 13 bytes, fits in one MSS=100 segment.
        segments = conn.segment_data(data)
        self.assertEqual(len(segments), 1)
        # Last (only) segment should have PSH|ACK.
        flags = segments[0][13]
        self.assertTrue(flags & PSH)
        self.assertTrue(flags & ACK)
        # Payload is the data.
        payload = segments[0][20:]  # 20-byte header, no options.
        self.assertEqual(payload, data)

    def test_segment_data_multiple_chunks(self):
        conn = self._make_conn(isn=100)
        conn.build_syn()
        conn.receive_syn_ack(2000)
        conn.build_ack()
        data = b"A" * 250  # 250 bytes, MSS=100 -> 3 segments.
        segments = conn.segment_data(data)
        self.assertEqual(len(segments), 3)
        # First two: ACK only. Last: PSH|ACK.
        self.assertTrue(segments[0][13] & ACK)
        self.assertFalse(segments[0][13] & PSH)
        self.assertTrue(segments[2][13] & PSH)
        self.assertTrue(segments[2][13] & ACK)

    def test_fin_transitions(self):
        conn = self._make_conn()
        conn.build_syn()
        conn.receive_syn_ack(2000)
        conn.build_ack()
        fin = conn.build_fin()
        self.assertEqual(conn.state, TCPState.FIN_WAIT_1)
        flags = fin[13]
        self.assertTrue(flags & FIN)

    def test_rst_transitions_to_closed(self):
        conn = self._make_conn()
        conn.build_syn()
        conn.receive_syn_ack(2000)
        conn.build_ack()
        conn.build_rst()
        self.assertEqual(conn.state, TCPState.CLOSED)

    def test_reserved_bits_always_zero(self):
        """Reserved bits in the TCP header must always be zero (MRF)."""
        conn = self._make_conn()
        syn = conn.build_syn()
        # Byte 12 upper nibble is data offset. Lower nibble + byte 13 upper 2 bits are reserved.
        # In our implementation, byte 12 = (data_offset << 4), so lower nibble = 0.
        self.assertEqual(syn[12] & 0x0F, 0)

    def test_urgent_pointer_always_zero(self):
        """Urgent pointer must always be zero (MRF)."""
        conn = self._make_conn()
        syn = conn.build_syn()
        # Urgent pointer is bytes 18-19 (but with MSS option, header is 24 bytes).
        urg_ptr = struct.unpack("!H", syn[18:20])[0]
        self.assertEqual(urg_ptr, 0)

    def test_determinism(self):
        """Two connections with same ISN produce identical segments."""
        conn1 = self._make_conn(isn=42)
        conn2 = self._make_conn(isn=42)
        syn1 = conn1.build_syn()
        syn2 = conn2.build_syn()
        self.assertEqual(syn1, syn2)

        conn1.receive_syn_ack(100)
        conn2.receive_syn_ack(100)
        ack1 = conn1.build_ack()
        ack2 = conn2.build_ack()
        self.assertEqual(ack1, ack2)

        data = b"deterministic payload"
        segs1 = conn1.segment_data(data)
        segs2 = conn2.segment_data(data)
        self.assertEqual(segs1, segs2)


if __name__ == "__main__":
    unittest.main()
