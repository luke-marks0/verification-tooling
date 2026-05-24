"""Deterministic TCP (L4) state machine and segment builder.

MRF policy:
  - ISN derived from sha256(run_id + connection_index)
  - No timestamps (RFC 7323)
  - No SACK
  - No window scaling
  - No Nagle algorithm (immediate send)
  - urgent_ptr=0, reserved bits=0
  - MSS option only in SYN
  - Software checksum (no offload)
  - Fixed window size from manifest ring_sizes.rx
"""
from __future__ import annotations

import hashlib
import struct
from enum import Enum, auto

from modules.network.networkdet.checksums import tcp_checksum


TCP_HEADER_LEN = 20  # Without options.
TCP_HEADER_LEN_WITH_MSS = 24  # With 4-byte MSS option.

# TCP flags.
FIN = 0x01
SYN = 0x02
RST = 0x04
PSH = 0x08
ACK = 0x10
URG = 0x20


class TCPState(Enum):
    CLOSED = auto()
    SYN_SENT = auto()
    SYN_RECEIVED = auto()
    ESTABLISHED = auto()
    FIN_WAIT_1 = auto()
    FIN_WAIT_2 = auto()
    CLOSE_WAIT = auto()
    LAST_ACK = auto()
    TIME_WAIT = auto()


def deterministic_isn(run_id: str, conn_index: int) -> int:
    """Derive a deterministic Initial Sequence Number.

    Uses SHA-256 of the run ID and connection index, then takes the
    first 4 bytes as a 32-bit unsigned integer.
    """
    digest = hashlib.sha256(f"{run_id}:{conn_index}".encode("utf-8")).digest()
    return struct.unpack("!I", digest[:4])[0]


class DeterministicTCPConnection:
    """Deterministic TCP connection state machine.

    Builds TCP segments with fully deterministic header fields.
    Does not implement retransmission or congestion control — the sim
    backend is reliable and the DPDK backend uses local delivery.
    """

    def __init__(
        self,
        src_port: int,
        dst_port: int,
        *,
        isn: int,
        mss: int = 1460,
        window: int = 65535,
        src_ip: bytes,
        dst_ip: bytes,
    ) -> None:
        self._src_port = src_port
        self._dst_port = dst_port
        self._mss = mss
        self._window = min(window, 65535)  # No window scaling.
        self._src_ip = src_ip
        self._dst_ip = dst_ip

        self._seq = isn
        self._ack = 0
        self._state = TCPState.CLOSED

    @property
    def state(self) -> TCPState:
        return self._state

    @property
    def seq(self) -> int:
        return self._seq & 0xFFFFFFFF

    @property
    def ack(self) -> int:
        return self._ack & 0xFFFFFFFF

    def _build_segment(
        self,
        flags: int,
        payload: bytes = b"",
        *,
        include_mss: bool = False,
    ) -> bytes:
        """Build a TCP segment with deterministic header fields."""
        if include_mss:
            data_offset = 6  # 24 bytes / 4
            options = struct.pack("!BBH", 2, 4, self._mss)  # MSS option
        else:
            data_offset = 5  # 20 bytes / 4
            options = b""

        # Build header with checksum zeroed.
        header = struct.pack(
            "!HHIIBBHHH",
            self._src_port,
            self._dst_port,
            self.seq,
            self.ack,
            (data_offset << 4),  # data offset + reserved (0)
            flags,
            self._window,
            0x0000,   # checksum placeholder
            0x0000,   # urgent pointer (always 0)
        )

        segment_no_cksum = header + options + payload
        cksum = tcp_checksum(self._src_ip, self._dst_ip, segment_no_cksum)

        # Rebuild with computed checksum.
        header = struct.pack(
            "!HHIIBBHHH",
            self._src_port,
            self._dst_port,
            self.seq,
            self.ack,
            (data_offset << 4),
            flags,
            self._window,
            cksum,
            0x0000,
        )

        return header + options + payload

    def build_syn(self) -> bytes:
        """Build a SYN segment (connection initiation)."""
        segment = self._build_segment(SYN, include_mss=True)
        self._seq += 1  # SYN consumes one sequence number.
        self._state = TCPState.SYN_SENT
        return segment

    def build_syn_ack(self, peer_isn: int) -> bytes:
        """Build a SYN-ACK segment (connection response)."""
        self._ack = (peer_isn + 1) & 0xFFFFFFFF
        segment = self._build_segment(SYN | ACK, include_mss=True)
        self._seq += 1
        self._state = TCPState.SYN_RECEIVED
        return segment

    def build_ack(self) -> bytes:
        """Build a bare ACK segment."""
        segment = self._build_segment(ACK)
        if self._state == TCPState.SYN_SENT:
            self._state = TCPState.ESTABLISHED
        return segment

    def receive_syn_ack(self, peer_seq: int) -> None:
        """Process a received SYN-ACK (update ack number)."""
        self._ack = (peer_seq + 1) & 0xFFFFFFFF
        self._state = TCPState.ESTABLISHED

    def receive_ack(self, _peer_ack: int) -> None:
        """Process a received ACK."""
        if self._state == TCPState.SYN_RECEIVED:
            self._state = TCPState.ESTABLISHED
        elif self._state == TCPState.FIN_WAIT_1:
            self._state = TCPState.FIN_WAIT_2

    def receive_data(self, data_len: int) -> None:
        """Update ack number after receiving data."""
        self._ack = (self._ack + data_len) & 0xFFFFFFFF

    def segment_data(self, data: bytes) -> list[bytes]:
        """Segment *data* into MSS-sized TCP data segments.

        Returns a list of complete TCP segments (header + payload) with
        PSH|ACK flags.  The last segment gets PSH set.
        """
        segments: list[bytes] = []
        offset = 0
        while offset < len(data):
            chunk = data[offset : offset + self._mss]
            is_last = offset + len(chunk) >= len(data)
            flags = PSH | ACK if is_last else ACK
            segment = self._build_segment(flags, chunk)
            self._seq = (self._seq + len(chunk)) & 0xFFFFFFFF
            segments.append(segment)
            offset += len(chunk)
        return segments

    def build_fin(self) -> bytes:
        """Build a FIN segment (connection teardown)."""
        segment = self._build_segment(FIN | ACK)
        self._seq += 1  # FIN consumes one sequence number.
        self._state = TCPState.FIN_WAIT_1
        return segment

    def build_rst(self) -> bytes:
        """Build a RST segment (connection reset, no payload per MRF)."""
        segment = self._build_segment(RST | ACK)
        self._state = TCPState.CLOSED
        return segment
