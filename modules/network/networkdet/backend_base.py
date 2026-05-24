"""Abstract network backend interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

from modules.network.networkdet.config import NetStackConfig
from modules.network.networkdet.tx_report import TxReport


class NetworkBackend(ABC):
    """Base class for network backends (sim and dpdk)."""

    @abstractmethod
    def init(self, config: NetStackConfig) -> None:
        """Initialise the backend with the given configuration."""

    @abstractmethod
    def send_frame(self, frame: bytes) -> None:
        """Transmit a single L2 frame."""

    @abstractmethod
    def recv_frame(self) -> bytes | None:
        """Receive a single L2 frame, or None if nothing is available."""

    @abstractmethod
    def close(self) -> None:
        """Tear down the backend and release resources."""

    def flush(self) -> TxReport | None:
        """Flush pending transmissions and return a TxReport, if supported."""
        return None
