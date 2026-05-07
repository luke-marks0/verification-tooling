"""Deterministic seed -> matrix bytes.

Cross-implementation portable: both stdlib and torch backends must produce
identical bytes for the same ``(seed, dtype, rows, cols)`` triple. This is
the contract that lets a stdlib verifier compare digests against a torch
prover (and vice versa).

Strategy:
  1. Expand ``(seed, dtype, rows, cols)`` to a deterministic byte stream
     using SHAKE-256 (stdlib).
  2. For each element, twiddle the bits so the value is a finite, bounded
     float (or a signed integer of the dtype). The bit-twiddle is a fixed,
     dtype-local mask + OR with a biased exponent that pins magnitudes
     to roughly ``[0.5, 1.0)`` in absolute value for floats.

The post-twiddle bytes are the canonical bytes of the matrix at the dtype:
SHA-256 of (rows * cols * bytes_per_elem) bytes is the digest.

Stdlib backend supports {int8, int32, fp64}; the other dtypes return raw
bytes that a torch backend can interpret. Pure-Python ``read_matrix`` only
handles the stdlib subset.
"""
from __future__ import annotations

import hashlib
import struct
from typing import Iterable

from pkg.freivalds.spec import SUPPORTED_DTYPES


_BYTES_PER_ELEM: dict[str, int] = {
    "int8": 1,
    "int32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "fp64": 8,
    "fp8_e4m3": 1,
}


def bytes_per_elem(dtype: str) -> int:
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    return _BYTES_PER_ELEM[dtype]


def _shake_bytes(seed: int, dtype: str, rows: int, cols: int, n_bytes: int) -> bytes:
    """Expand (seed, dtype, rows, cols) into ``n_bytes`` deterministic bytes."""
    label = f"freivalds-prng-v1|{dtype}|{rows}x{cols}|{seed}".encode("utf-8")
    return hashlib.shake_256(label).digest(n_bytes)


try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _np = None
    _HAS_NUMPY = False


def _twiddle_fp64(buf: bytes) -> bytes:
    """Force biased-exponent 1022; magnitude ends up in ``[0.5, 1.0)``."""
    if _HAS_NUMPY:
        arr = _np.frombuffer(buf, dtype="<u8").copy()
        arr &= _np.uint64(0x800FFFFFFFFFFFFF)
        arr |= _np.uint64(1022) << _np.uint64(52)
        return arr.tobytes()
    out = bytearray(len(buf))
    for i in range(0, len(buf), 8):
        (val,) = struct.unpack_from("<Q", buf, i)
        val = (val & 0x800FFFFFFFFFFFFF) | (1022 << 52)
        struct.pack_into("<Q", out, i, val)
    return bytes(out)


def _twiddle_fp32(buf: bytes) -> bytes:
    if _HAS_NUMPY:
        arr = _np.frombuffer(buf, dtype="<u4").copy()
        arr &= _np.uint32(0x807FFFFF)
        arr |= _np.uint32(126) << _np.uint32(23)
        return arr.tobytes()
    out = bytearray(len(buf))
    for i in range(0, len(buf), 4):
        (val,) = struct.unpack_from("<I", buf, i)
        val = (val & 0x807FFFFF) | (126 << 23)
        struct.pack_into("<I", out, i, val)
    return bytes(out)


def _twiddle_bf16(buf: bytes) -> bytes:
    if _HAS_NUMPY:
        arr = _np.frombuffer(buf, dtype="<u2").copy()
        arr &= _np.uint16(0x807F)
        arr |= _np.uint16(126) << _np.uint16(7)
        return arr.tobytes()
    out = bytearray(len(buf))
    for i in range(0, len(buf), 2):
        (val,) = struct.unpack_from("<H", buf, i)
        val = (val & 0x807F) | (126 << 7)
        struct.pack_into("<H", out, i, val)
    return bytes(out)


def _twiddle_fp16(buf: bytes) -> bytes:
    if _HAS_NUMPY:
        arr = _np.frombuffer(buf, dtype="<u2").copy()
        arr &= _np.uint16(0x83FF)
        arr |= _np.uint16(14) << _np.uint16(10)
        return arr.tobytes()
    out = bytearray(len(buf))
    for i in range(0, len(buf), 2):
        (val,) = struct.unpack_from("<H", buf, i)
        val = (val & 0x83FF) | (14 << 10)
        struct.pack_into("<H", out, i, val)
    return bytes(out)


def _twiddle_fp8_e4m3(buf: bytes) -> bytes:
    if _HAS_NUMPY:
        arr = _np.frombuffer(buf, dtype=_np.uint8).copy()
        arr &= _np.uint8(0x87)
        arr |= _np.uint8(6) << _np.uint8(3)
        return arr.tobytes()
    return bytes(((b & 0x87) | (6 << 3)) for b in buf)


# Integers are used as-is from the SHAKE stream; sign comes from the high bit.
def _twiddle_int(buf: bytes) -> bytes:
    return buf


_TWIDDLERS = {
    "int8": _twiddle_int,
    "int32": _twiddle_int,
    "fp16": _twiddle_fp16,
    "bf16": _twiddle_bf16,
    "fp32": _twiddle_fp32,
    "fp64": _twiddle_fp64,
    "fp8_e4m3": _twiddle_fp8_e4m3,
}


def gen_matrix_bytes(seed: int, dtype: str, rows: int, cols: int) -> bytes:
    """Return canonical row-major bytes for the matrix.

    Length: ``rows * cols * bytes_per_elem(dtype)``. Identical across any
    backend that follows this spec.
    """
    n_bytes = rows * cols * bytes_per_elem(dtype)
    raw = _shake_bytes(seed, dtype, rows, cols, n_bytes)
    return _TWIDDLERS[dtype](raw)


def matrix_digest(canonical_bytes: bytes) -> str:
    """``sha256:`` prefixed digest of canonical matrix bytes."""
    return f"sha256:{hashlib.sha256(canonical_bytes).hexdigest()}"


# --- stdlib readers -------------------------------------------------------
# Only int8, int32, fp64 are interpretable in pure stdlib. The other dtypes
# need numpy / torch to materialise as numbers.

def read_matrix_int8(buf: bytes, rows: int, cols: int) -> list[list[int]]:
    if len(buf) != rows * cols:
        raise ValueError(f"buf len {len(buf)} != rows*cols {rows*cols}")
    out: list[list[int]] = []
    for r in range(rows):
        row = list(struct.unpack_from(f"<{cols}b", buf, r * cols))
        out.append(row)
    return out


def read_matrix_int32(buf: bytes, rows: int, cols: int) -> list[list[int]]:
    if len(buf) != rows * cols * 4:
        raise ValueError(f"buf len {len(buf)} != rows*cols*4 {rows*cols*4}")
    out: list[list[int]] = []
    stride = cols * 4
    for r in range(rows):
        row = list(struct.unpack_from(f"<{cols}i", buf, r * stride))
        out.append(row)
    return out


def read_matrix_fp64(buf: bytes, rows: int, cols: int) -> list[list[float]]:
    if len(buf) != rows * cols * 8:
        raise ValueError(f"buf len {len(buf)} != rows*cols*8 {rows*cols*8}")
    out: list[list[float]] = []
    stride = cols * 8
    for r in range(rows):
        row = list(struct.unpack_from(f"<{cols}d", buf, r * stride))
        out.append(row)
    return out


_STDLIB_READERS = {
    "int8": read_matrix_int8,
    "int32": read_matrix_int32,
    "fp64": read_matrix_fp64,
}


def is_stdlib_supported(dtype: str) -> bool:
    return dtype in _STDLIB_READERS


def read_matrix_stdlib(buf: bytes, dtype: str, rows: int, cols: int) -> list[list]:
    if dtype not in _STDLIB_READERS:
        raise ValueError(
            f"dtype {dtype!r} not supported by stdlib backend; "
            f"available: {sorted(_STDLIB_READERS)}"
        )
    return _STDLIB_READERS[dtype](buf, rows, cols)


def gen_matrix_stdlib(seed: int, dtype: str, rows: int, cols: int) -> tuple[list[list], bytes]:
    """Return ``(matrix, canonical_bytes)`` for stdlib-supported dtypes."""
    canonical = gen_matrix_bytes(seed, dtype, rows, cols)
    matrix = read_matrix_stdlib(canonical, dtype, rows, cols)
    return matrix, canonical


def write_matrix_bytes_stdlib(matrix: Iterable[Iterable], dtype: str) -> bytes:
    """Pack a stdlib matrix into canonical bytes."""
    if dtype == "int8":
        out = bytearray()
        for row in matrix:
            for v in row:
                out.extend(struct.pack("<b", int(v)))
        return bytes(out)
    if dtype == "int32":
        out = bytearray()
        for row in matrix:
            for v in row:
                out.extend(struct.pack("<i", int(v)))
        return bytes(out)
    if dtype == "fp64":
        out = bytearray()
        for row in matrix:
            for v in row:
                out.extend(struct.pack("<d", float(v)))
        return bytes(out)
    raise ValueError(
        f"dtype {dtype!r} not supported by stdlib backend; available: int8, int32, fp64"
    )
