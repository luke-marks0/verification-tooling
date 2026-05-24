"""DPDK network backend for real NIC transmission."""
from __future__ import annotations

import hashlib

from modules.network.networkdet.backend_base import NetworkBackend
from modules.network.networkdet.config import NetStackConfig
from modules.network.networkdet.tx_report import TxReport


class DPDKBackend(NetworkBackend):
    """Kernel-bypass backend via DPDK + libnetdet.so.

    Frames are buffered in Python and flushed to the NIC in a single
    burst via flush(). The simulated backend sends immediately in
    send_frame(); this backend batches for efficiency and to compute
    a single TX completion digest over the entire run.
    """

    def __init__(self, *, port_id: int = 0, eal_args: list[str] | None = None):
        self._port_id = port_id
        self._eal_args = eal_args or []
        self._ctx: int | None = None
        self._lib = None
        self._tx_buffer: list[bytes] = []
        self._initialised = False

    def init(self, config: NetStackConfig) -> None:
        if config.tso or config.gso or config.checksum_offload:
            raise RuntimeError(
                "DPDK backend requires all offloads disabled. "
                "Got tso=%s gso=%s checksum_offload=%s"
                % (config.tso, config.gso, config.checksum_offload)
            )
        # Lazy import — only load libnetdet when DPDK is actually used
        from modules.network.networkdet.libnetdet_ffi import LibNetDet
        self._lib = LibNetDet()
        self._ctx = self._lib.init(self._eal_args, self._port_id)
        self._tx_buffer.clear()
        self._initialised = True

    def send_frame(self, frame: bytes) -> None:
        if not self._initialised:
            raise RuntimeError("DPDKBackend not initialised")
        self._tx_buffer.append(bytes(frame))

    def recv_frame(self) -> bytes | None:
        # For loopback verification, use recv_loopback() instead.
        # This method exists to satisfy the interface.
        return None

    def flush(self) -> TxReport | None:
        """Transmit all buffered frames and return an egress report."""
        if not self._initialised or not self._tx_buffer:
            return None

        result = self._lib.send(self._ctx, self._tx_buffer)

        # Compute pre-enqueue digest over only the confirmed frames,
        # so it matches the tx_completion digest scope (which covers
        # only what rte_eth_tx_burst accepted).
        h = hashlib.sha256()
        for frame in self._tx_buffer[:result.confirmed]:
            h.update(frame)
        pre_enqueue = f"sha256:{h.hexdigest()}"

        self._tx_buffer.clear()

        return TxReport(
            pre_enqueue_digest=pre_enqueue,
            tx_completion_digest=result.digest_prefixed,
            frames_submitted=result.submitted,
            frames_confirmed=result.confirmed,
        )

    def recv_loopback(self, timeout_ms: int = 1000) -> tuple[list[bytes], str]:
        """Receive frames for loopback verification.

        Returns (frames, digest) where digest is sha256:<hex>.
        """
        if not self._initialised:
            raise RuntimeError("DPDKBackend not initialised")
        result = self._lib.recv(self._ctx, timeout_ms)
        # Extract frame bytes before freeing
        frames = []
        for i in range(result.count):
            length = result.lengths[i]
            frame = bytes(result.frames[i][:length])
            frames.append(frame)
        digest = result.digest_prefixed
        self._lib.rx_free(result)
        return frames, digest

    def close(self) -> None:
        if self._ctx is not None:
            self._lib.close(self._ctx)
            self._ctx = None
        self._tx_buffer.clear()
        self._initialised = False
