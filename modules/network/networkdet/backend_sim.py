"""Simulated network backend for testing without hardware.

All transmitted frames are placed into an in-memory receive queue
(loopback).  Deterministic by construction — no real I/O, no kernel
involvement, no timing dependencies.
"""
from __future__ import annotations

from collections import deque

from modules.network.networkdet.backend_base import NetworkBackend
from modules.network.networkdet.config import NetStackConfig


class SimulatedBackend(NetworkBackend):
    """In-memory loopback backend.

    Sent frames appear in the receive queue in FIFO order.
    """

    def __init__(self) -> None:
        self._rx_queue: deque[bytes] = deque()
        self._initialised = False

    def init(self, config: NetStackConfig) -> None:
        self._rx_queue.clear()
        self._initialised = True

    def send_frame(self, frame: bytes) -> None:
        if not self._initialised:
            raise RuntimeError("SimulatedBackend not initialised")
        self._rx_queue.append(bytes(frame))

    def recv_frame(self) -> bytes | None:
        if not self._initialised:
            raise RuntimeError("SimulatedBackend not initialised")
        if self._rx_queue:
            return self._rx_queue.popleft()
        return None

    def close(self) -> None:
        self._rx_queue.clear()
        self._initialised = False
