"""Transmission report for egress integrity verification."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TxReport:
    """Result of transmitting frames through a real backend.

    Both digests use the ``sha256:<hex>`` format used elsewhere in
    the codebase (see CaptureRing.digest()).
    """

    pre_enqueue_digest: str
    tx_completion_digest: str
    frames_submitted: int
    frames_confirmed: int
    rx_loopback_digest: str | None = None
    rx_loopback_count: int | None = None

    @property
    def match(self) -> bool:
        base = self.pre_enqueue_digest == self.tx_completion_digest
        if self.rx_loopback_digest is not None:
            return base and self.tx_completion_digest == self.rx_loopback_digest
        return base

    @property
    def level(self) -> str:
        return "loopback" if self.rx_loopback_digest else "tx_completion"
