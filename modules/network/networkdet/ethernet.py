"""Deterministic Ethernet (L2) frame construction.

Every field is fixed or derived deterministically from configuration.
No VLAN tags, no padding randomisation, no jumbo frame negotiation.
"""
from __future__ import annotations

import struct


# Standard Ethernet header: dst(6) + src(6) + ethertype(2) = 14 bytes
ETHERNET_HEADER_LEN = 14
ETHERTYPE_IPV4 = 0x0800
# Minimum Ethernet payload (header excluded): 46 bytes
MIN_ETHERNET_PAYLOAD = 46


def mac_to_bytes(mac_str: str) -> bytes:
    """Convert a colon-separated MAC address string to 6 bytes."""
    parts = mac_str.split(":")
    if len(parts) != 6:
        raise ValueError(f"Invalid MAC address: {mac_str}")
    return bytes(int(p, 16) for p in parts)


def build_ethernet_frame(
    dst_mac: bytes,
    src_mac: bytes,
    payload: bytes,
    *,
    ethertype: int = ETHERTYPE_IPV4,
) -> bytes:
    """Build a deterministic Ethernet frame.

    The frame consists of:
      - 6-byte destination MAC
      - 6-byte source MAC
      - 2-byte EtherType
      - payload (zero-padded to 46 bytes if shorter)

    No FCS is appended — hardware computes it on transmit, and the sim
    backend does not require it.
    """
    header = struct.pack("!6s6sH", dst_mac, src_mac, ethertype)
    # Pad payload to minimum Ethernet payload length with deterministic zeros.
    if len(payload) < MIN_ETHERNET_PAYLOAD:
        payload = payload + b"\x00" * (MIN_ETHERNET_PAYLOAD - len(payload))
    return header + payload
