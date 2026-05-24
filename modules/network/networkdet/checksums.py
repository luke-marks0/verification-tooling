"""Pure-Python IP and TCP checksum computation (RFC 1071).

These implementations produce bitwise-identical results to hardware
checksum offload engines.  We compute checksums in software because
hardware offload introduces NIC-firmware-dependent nondeterminism.
"""
from __future__ import annotations

import struct


def _ones_complement_sum(data: bytes) -> int:
    """Compute the ones-complement sum of *data* treated as 16-bit words.

    If *data* has an odd length, it is padded with a zero byte on the right.
    """
    if len(data) % 2:
        data = data + b"\x00"

    total = 0
    for i in range(0, len(data), 2):
        word = (data[i] << 8) | data[i + 1]
        total += word
        # Fold carry bits back into the 16-bit accumulator.
        total = (total & 0xFFFF) + (total >> 16)

    return total


def ip_checksum(header_bytes: bytes) -> int:
    """Return the IPv4 header checksum as a 16-bit integer.

    *header_bytes* must be the complete IPv4 header with the checksum
    field set to zero.
    """
    return (~_ones_complement_sum(header_bytes)) & 0xFFFF


def tcp_checksum(src_ip: bytes, dst_ip: bytes, tcp_segment: bytes) -> int:
    """Return the TCP checksum as a 16-bit integer.

    *src_ip* and *dst_ip* are 4-byte packed IPv4 addresses.
    *tcp_segment* is the full TCP segment (header + payload) with the
    checksum field set to zero.
    """
    # Build the TCP pseudo-header (RFC 793 section 3.1).
    tcp_length = len(tcp_segment)
    pseudo_header = struct.pack(
        "!4s4sBBH",
        src_ip,
        dst_ip,
        0,        # reserved
        6,        # protocol (TCP)
        tcp_length,
    )
    return (~_ones_complement_sum(pseudo_header + tcp_segment)) & 0xFFFF
