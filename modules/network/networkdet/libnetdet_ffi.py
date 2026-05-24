"""ctypes bindings for libnetdet.so (DPDK transmit/receive)."""
from __future__ import annotations

import ctypes
import os
from ctypes import (
    POINTER, Structure, c_char_p, c_int, c_uint8, c_uint16, c_void_p,
)
from pathlib import Path


class TxResult(Structure):
    _fields_ = [
        ("confirmed", c_int),
        ("submitted", c_int),
        ("digest", c_uint8 * 32),
    ]

    @property
    def digest_hex(self) -> str:
        return bytes(self.digest).hex()

    @property
    def digest_prefixed(self) -> str:
        return f"sha256:{self.digest_hex}"


class RxResult(Structure):
    _fields_ = [
        ("count", c_int),
        ("frames", POINTER(POINTER(c_uint8))),
        ("lengths", POINTER(c_uint16)),
        ("digest", c_uint8 * 32),
    ]

    @property
    def digest_prefixed(self) -> str:
        return f"sha256:{bytes(self.digest).hex()}"


def _find_library() -> str:
    """Find libnetdet.so. Search order:

    1. LIBNETDET_PATH environment variable
    2. modules/network/native/libnetdet/build/libnetdet.so (development)
    3. /usr/local/lib/libnetdet.so (installed)
    """
    env_path = os.environ.get("LIBNETDET_PATH")
    if env_path:
        return env_path

    dev_path = Path(__file__).resolve().parents[1] / "native/libnetdet/build/libnetdet.so"
    if dev_path.exists():
        return str(dev_path)

    return "libnetdet.so"  # Let ctypes search LD_LIBRARY_PATH


class LibNetDet:
    """Wrapper around libnetdet.so."""

    def __init__(self, lib_path: str | None = None):
        path = lib_path or _find_library()
        self._lib = ctypes.CDLL(path)
        self._setup_signatures()

    def _setup_signatures(self):
        # netdet_init
        self._lib.netdet_init.argtypes = [c_int, POINTER(c_char_p), c_uint16]
        self._lib.netdet_init.restype = c_void_p

        # netdet_send
        self._lib.netdet_send.argtypes = [
            c_void_p,
            POINTER(POINTER(c_uint8)),
            POINTER(c_uint16),
            c_int,
        ]
        self._lib.netdet_send.restype = TxResult

        # netdet_recv
        self._lib.netdet_recv.argtypes = [c_void_p, c_int]
        self._lib.netdet_recv.restype = RxResult

        # netdet_rx_free
        self._lib.netdet_rx_free.argtypes = [POINTER(RxResult)]
        self._lib.netdet_rx_free.restype = None

        # netdet_close
        self._lib.netdet_close.argtypes = [c_void_p]
        self._lib.netdet_close.restype = None

    def init(self, eal_args: list[str], port_id: int) -> int:
        """Initialize DPDK. Returns an opaque context handle (as int)."""
        argc = len(eal_args) + 1
        argv_type = c_char_p * argc
        argv = argv_type(b"netdet", *[a.encode() for a in eal_args])
        ctx = self._lib.netdet_init(argc, argv, port_id)
        if not ctx:
            raise RuntimeError("netdet_init failed — check DPDK EAL args and port ID")
        return ctx

    def send(self, ctx: int, frames: list[bytes]) -> TxResult:
        """Send pre-built L2 frames. Returns TX result with digest."""
        count = len(frames)
        frame_ptrs = (POINTER(c_uint8) * count)()
        lengths = (c_uint16 * count)()
        # Keep references to prevent GC
        buffers = []
        for i, frame in enumerate(frames):
            buf = (c_uint8 * len(frame))(*frame)
            buffers.append(buf)
            frame_ptrs[i] = ctypes.cast(buf, POINTER(c_uint8))
            lengths[i] = len(frame)
        return self._lib.netdet_send(ctx, frame_ptrs, lengths, count)

    def recv(self, ctx: int, timeout_ms: int = 1000) -> RxResult:
        """Receive frames (for loopback verification)."""
        return self._lib.netdet_recv(ctx, timeout_ms)

    def rx_free(self, result: RxResult) -> None:
        """Free RX result buffers."""
        self._lib.netdet_rx_free(ctypes.byref(result))

    def close(self, ctx: int) -> None:
        """Shut down DPDK port and EAL."""
        self._lib.netdet_close(ctx)
