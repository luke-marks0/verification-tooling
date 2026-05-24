"""Unit tests for deterministic IP and TCP checksum computation."""
from __future__ import annotations

import struct
import unittest

from modules.network.networkdet.checksums import ip_checksum, tcp_checksum


class TestIPChecksum(unittest.TestCase):
    """Verify IP checksum against known-good values."""

    def test_rfc1071_example(self):
        """RFC 1071 section 3 example: sum of 0x0001, 0xF203, ..."""
        # Example from RFC 1071: the checksum of these words should
        # produce a valid zero-check when the checksum is included.
        data = bytes([
            0x45, 0x00, 0x00, 0x73,  # ver/ihl, tos, total_len
            0x00, 0x00, 0x40, 0x00,  # id, flags/frag
            0x40, 0x11, 0x00, 0x00,  # ttl, proto(UDP), cksum=0
            0xC0, 0xA8, 0x00, 0x01,  # src: 192.168.0.1
            0xC0, 0xA8, 0x00, 0xC7,  # dst: 192.168.0.199
        ])
        cksum = ip_checksum(data)
        # Verify checksum is nonzero.
        self.assertNotEqual(cksum, 0)
        # Insert checksum and verify the full header checksums to zero.
        header_with_cksum = data[:10] + struct.pack("!H", cksum) + data[12:]
        self.assertEqual(ip_checksum(header_with_cksum), 0)

    def test_all_zeros(self):
        """Checksum of 20 zero bytes should be 0xFFFF."""
        data = b"\x00" * 20
        self.assertEqual(ip_checksum(data), 0xFFFF)

    def test_determinism(self):
        """Same input always produces the same checksum."""
        data = bytes(range(20))
        c1 = ip_checksum(data)
        c2 = ip_checksum(data)
        self.assertEqual(c1, c2)


class TestTCPChecksum(unittest.TestCase):
    """Verify TCP checksum with pseudo-header."""

    def test_basic_segment(self):
        """Verify TCP checksum for a minimal SYN segment."""
        src_ip = bytes([10, 0, 0, 1])
        dst_ip = bytes([10, 0, 0, 2])
        # Minimal TCP header (20 bytes), all fields zero except checksum.
        tcp_header = b"\x00" * 20
        cksum = tcp_checksum(src_ip, dst_ip, tcp_header)
        self.assertIsInstance(cksum, int)
        self.assertTrue(0 <= cksum <= 0xFFFF)

    def test_checksum_validates_to_zero(self):
        """Insert the computed checksum and verify the segment validates."""
        src_ip = bytes([192, 168, 1, 10])
        dst_ip = bytes([192, 168, 1, 20])
        # Build a segment with checksum field zeroed (bytes 16-17).
        tcp_no_cksum = struct.pack(
            "!HHIIBBHHH",
            12345,  # src port
            80,     # dst port
            100,    # seq
            0,      # ack
            0x50,   # data offset = 5 (20 bytes)
            0x02,   # SYN
            65535,  # window
            0,      # checksum = 0
            0,      # urgent ptr
        )
        cksum = tcp_checksum(src_ip, dst_ip, tcp_no_cksum)
        # Insert checksum.
        tcp_with_cksum = tcp_no_cksum[:16] + struct.pack("!H", cksum) + tcp_no_cksum[18:]
        # Recompute — should be zero.
        self.assertEqual(tcp_checksum(src_ip, dst_ip, tcp_with_cksum), 0)

    def test_determinism(self):
        """Same inputs always produce the same checksum."""
        src_ip = bytes([10, 0, 0, 1])
        dst_ip = bytes([10, 0, 0, 2])
        segment = bytes(range(20))
        c1 = tcp_checksum(src_ip, dst_ip, segment)
        c2 = tcp_checksum(src_ip, dst_ip, segment)
        self.assertEqual(c1, c2)


if __name__ == "__main__":
    unittest.main()
