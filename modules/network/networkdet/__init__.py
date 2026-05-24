"""Deterministic userspace networking stack.

Public API for building bitwise-reproducible L2 frames from
application-layer data.  See ADR-0004 for architecture rationale.

Usage::

    from modules.network.networkdet import create_net_stack

    net = create_net_stack(manifest, lockfile, backend="sim")
    frames = net.process_response(conn_index=0, response_bytes=b"...")
    digest = net.capture_digest()
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from modules.network.networkdet.backend_base import NetworkBackend
from modules.network.networkdet.backend_sim import SimulatedBackend
from modules.network.networkdet.capture import CaptureRing
from modules.network.networkdet.config import NetStackConfig, parse_net_config
from modules.network.networkdet.frame import DeterministicFrameBuilder
from modules.network.networkdet.tx_report import TxReport


class DeterministicNetStack:
    """Facade over the deterministic networking stack.

    Manages configuration, capture ring, backend, and per-connection
    frame builders.
    """

    def __init__(
        self,
        config: NetStackConfig,
        *,
        run_id: str,
        backend: NetworkBackend | None = None,
        verify_mrf: bool = False,
    ) -> None:
        self._config = config
        self._run_id = run_id
        self._capture = CaptureRing()
        self._backend = backend or SimulatedBackend()
        self._backend.init(config)
        self._builders: dict[int, DeterministicFrameBuilder] = {}
        self._warden = None
        if verify_mrf:
            from modules.network.networkdet.warden import ActiveWarden
            self._warden = ActiveWarden()

    @property
    def capture_ring(self) -> CaptureRing:
        return self._capture

    def _get_builder(self, conn_index: int) -> DeterministicFrameBuilder:
        if conn_index not in self._builders:
            self._builders[conn_index] = DeterministicFrameBuilder(
                self._config,
                run_id=self._run_id,
                conn_index=conn_index,
                capture_ring=self._capture,
            )
        return self._builders[conn_index]

    def process_response(
        self,
        conn_index: int,
        response_bytes: bytes,
    ) -> list[bytes]:
        """Build deterministic L2 frames for a response payload.

        Frames are recorded in the capture ring and optionally sent
        through the backend.  When verify_mrf is enabled, each frame
        is passed through the ActiveWarden and structural violations
        are asserted to be zero.
        """
        builder = self._get_builder(conn_index)
        frames = builder.build_response_frames(response_bytes)
        for i, frame in enumerate(frames):
            if self._warden is not None:
                self._warden.reset()
                self._warden.normalize(frame)
                s = self._warden.stats
                violations = (
                    s.reserved_bits_zeroed + s.urgent_ptr_zeroed
                    + s.options_stripped + s.timestamps_stripped
                    + s.rst_payloads_stripped + s.tos_zeroed
                    + s.ttl_normalized + s.padding_zeroed
                )
                if violations > 0:
                    raise RuntimeError(
                        f"Frame builder produced non-MRF-compliant frame "
                        f"(conn_index={conn_index}, frame={i}). "
                        f"Warden violations: {s.as_dict()}. "
                        f"This is a bug in the frame builder."
                    )
            self._backend.send_frame(frame)
        return frames

    def process_exchange(
        self,
        conn_index: int,
        request_bytes: bytes,
        response_bytes: bytes,
    ) -> list[bytes]:
        """Build deterministic L2 frames for a request/response exchange.

        The request bytes are used to build inbound frames (for the
        capture record), and the response bytes are segmented into
        outbound frames.  All frames are captured.
        """
        builder = self._get_builder(conn_index)
        all_frames: list[bytes] = []

        # Inbound request frames.
        req_frames = builder.build_data_frames(request_bytes)
        all_frames.extend(req_frames)

        # Outbound response frames.
        resp_frames = builder.build_response_frames(response_bytes)
        all_frames.extend(resp_frames)

        for frame in all_frames:
            self._backend.send_frame(frame)

        return all_frames

    def capture_digest(self) -> str:
        """SHA-256 digest over all captured frames."""
        return self._capture.digest()

    def capture_frames_hex(self) -> list[dict[str, object]]:
        """Return all captured frames as JSON-serializable dicts."""
        return self._capture.frames_as_hex()

    def frame_count(self) -> int:
        return self._capture.frame_count

    def flush(self) -> TxReport | None:
        """Flush the backend and return a TxReport, if supported."""
        return self._backend.flush()

    def close(self) -> None:
        self._backend.close()


def create_net_stack(
    manifest: dict[str, Any],
    lockfile: dict[str, Any],
    *,
    backend: str = "sim",
    run_id: str | None = None,
    src_ip: str = "10.0.0.1",
    dst_ip: str = "10.0.0.2",
    src_mac: str = "02:00:00:00:00:01",
    dst_mac: str = "02:00:00:00:00:02",
    src_port: int = 8000,
    dst_port: int = 80,
    **kwargs: Any,
) -> DeterministicNetStack:
    """Create a deterministic network stack from manifest and lockfile.

    Args:
        manifest: Parsed manifest JSON.
        lockfile: Parsed lockfile JSON (unused currently, reserved for
            artifact digest validation in Phase 4).
        backend: ``"sim"`` for simulated or ``"dpdk"`` for real NIC.
        run_id: Override run ID (defaults to manifest's ``run_id``).
        src_ip: Source IPv4 address for frame construction.
        dst_ip: Destination IPv4 address.
        src_mac: Source MAC address.
        dst_mac: Destination MAC address.
        src_port: TCP source port.
        dst_port: TCP destination port.
        **kwargs: Additional keyword arguments passed to the backend
            (e.g. ``dpdk_port``, ``dpdk_eal_args`` for the DPDK backend).
    """
    config = parse_net_config(manifest)
    config = replace(
        config,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_mac=src_mac,
        dst_mac=dst_mac,
        src_port=src_port,
        dst_port=dst_port,
    )

    effective_run_id = run_id or manifest.get("run_id", "default-run")

    if backend == "sim":
        be: NetworkBackend = SimulatedBackend()
    elif backend == "dpdk":
        from modules.network.networkdet.backend_dpdk import DPDKBackend
        dpdk_port = kwargs.get("dpdk_port", 0)
        dpdk_eal_args = kwargs.get("dpdk_eal_args", [])
        be = DPDKBackend(port_id=dpdk_port, eal_args=dpdk_eal_args)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    return DeterministicNetStack(
        config,
        run_id=effective_run_id,
        backend=be,
        verify_mrf=(backend == "dpdk"),
    )
