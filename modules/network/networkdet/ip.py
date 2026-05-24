"""Deterministic IPv4 (L3) packet construction.

Every header field is fixed or derived from a deterministic counter.
MRF policy:
  - version=4, IHL=5 (no options)
  - DSCP/ECN=0, TTL=64, DF=1, MF=0
  - IP ID: deterministic counter starting at 0
  - No fragmentation (MSS enforced at TCP layer)
  - Software checksum (no offload)
"""
from __future__ import annotations

import socket
import struct

from modules.network.networkdet.checksums import ip_checksum


# IPv4 header length in bytes (no options).
IPV4_HEADER_LEN = 20
# Default TTL for all packets.
DEFAULT_TTL = 64
# Protocol number for TCP.
PROTO_TCP = 6


def ip_to_bytes(ip_str: str) -> bytes:
    """Convert a dotted-decimal IPv4 address to 4 bytes."""
    return socket.inet_aton(ip_str)


class DeterministicIPLayer:
    """Deterministic IPv4 packet builder.

    The IP identification field uses a simple counter starting at 0,
    incremented by 1 for each packet.  This eliminates the entropy
    that kernel stacks inject via random or hash-based ID generation.
    """

    def __init__(self, src_ip: str, dst_ip: str, *, ttl: int = DEFAULT_TTL) -> None:
        self._src_ip = ip_to_bytes(src_ip)
        self._dst_ip = ip_to_bytes(dst_ip)
        self._ttl = ttl
        self._ip_id_counter = 0

    @property
    def src_ip(self) -> bytes:
        return self._src_ip

    @property
    def dst_ip(self) -> bytes:
        return self._dst_ip

    def build_packet(self, protocol: int, payload: bytes) -> bytes:
        """Build a complete IPv4 packet with deterministic header fields.

        Returns the full IP packet (header + payload).
        """
        total_length = IPV4_HEADER_LEN + len(payload)
        ip_id = self._ip_id_counter & 0xFFFF
        self._ip_id_counter += 1

        # Flags: DF=1, MF=0  ->  0x4000
        flags_fragment = 0x4000

        # Build header with checksum field zeroed.
        header_no_cksum = struct.pack(
            "!BBHHHBBH4s4s",
            0x45,            # version=4, IHL=5
            0x00,            # DSCP=0, ECN=0
            total_length,
            ip_id,
            flags_fragment,
            self._ttl,
            protocol,
            0x0000,          # checksum placeholder
            self._src_ip,
            self._dst_ip,
        )

        cksum = ip_checksum(header_no_cksum)

        # Rebuild with computed checksum.
        header = struct.pack(
            "!BBHHHBBH4s4s",
            0x45,
            0x00,
            total_length,
            ip_id,
            flags_fragment,
            self._ttl,
            protocol,
            cksum,
            self._src_ip,
            self._dst_ip,
        )

        return header + payload

    def reset(self) -> None:
        """Reset the IP ID counter (call between runs)."""
        self._ip_id_counter = 0
