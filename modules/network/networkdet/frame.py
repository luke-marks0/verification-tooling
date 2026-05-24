"""Deterministic L2+L3+L4 frame builder.

Composes Ethernet, IPv4, and TCP layers into complete L2 frames
that are bitwise-reproducible given the same inputs and connection state.
"""
from __future__ import annotations

from modules.network.networkdet.capture import CaptureRing
from modules.network.networkdet.config import NetStackConfig
from modules.network.networkdet.ethernet import build_ethernet_frame, mac_to_bytes
from modules.network.networkdet.ip import PROTO_TCP, DeterministicIPLayer
from modules.network.networkdet.tcp import DeterministicTCPConnection, deterministic_isn


class DeterministicFrameBuilder:
    """Builds complete L2 frames from application data.

    Each instance manages a single TCP connection's frame construction.
    Frames are recorded in the provided :class:`CaptureRing` before
    being returned.
    """

    def __init__(
        self,
        config: NetStackConfig,
        *,
        run_id: str,
        conn_index: int,
        capture_ring: CaptureRing,
    ) -> None:
        self._config = config
        self._capture = capture_ring
        self._src_mac = mac_to_bytes(config.src_mac)
        self._dst_mac = mac_to_bytes(config.dst_mac)

        self._ip = DeterministicIPLayer(config.src_ip, config.dst_ip)

        isn = deterministic_isn(run_id, conn_index)
        self._tcp = DeterministicTCPConnection(
            config.src_port,
            config.dst_port,
            isn=isn,
            mss=config.mss,
            window=config.ring_rx,
            src_ip=self._ip.src_ip,
            dst_ip=self._ip.dst_ip,
        )

    @property
    def tcp(self) -> DeterministicTCPConnection:
        return self._tcp

    def _wrap_frame(self, tcp_segment: bytes) -> bytes:
        """Wrap a TCP segment in IP and Ethernet headers."""
        ip_packet = self._ip.build_packet(PROTO_TCP, tcp_segment)
        frame = build_ethernet_frame(self._dst_mac, self._src_mac, ip_packet)
        self._capture.record(frame)
        return frame

    def build_syn(self) -> bytes:
        return self._wrap_frame(self._tcp.build_syn())

    def build_syn_ack(self, peer_isn: int) -> bytes:
        return self._wrap_frame(self._tcp.build_syn_ack(peer_isn))

    def build_ack(self) -> bytes:
        return self._wrap_frame(self._tcp.build_ack())

    def build_data_frames(self, data: bytes) -> list[bytes]:
        """Segment *data* and return a list of L2 frames."""
        segments = self._tcp.segment_data(data)
        return [self._wrap_frame(seg) for seg in segments]

    def build_fin(self) -> bytes:
        return self._wrap_frame(self._tcp.build_fin())

    def build_response_frames(self, response_bytes: bytes) -> list[bytes]:
        """Build all frames for a complete HTTP response.

        This is the primary entry point for the runner/capture pipeline.
        It segments the response data into MSS-sized chunks, wraps each
        in TCP/IP/Ethernet headers, and records all frames in the capture ring.
        """
        return self.build_data_frames(response_bytes)
