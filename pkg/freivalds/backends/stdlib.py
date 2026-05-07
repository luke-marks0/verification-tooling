"""Pure-Python backend for the Freivalds prover/verifier.

Supports ``int8``, ``int32``, ``fp64`` matrices. Slow (nested loops), but
useful for: (1) running unit tests on a CPU-only dev box without torch;
(2) acting as the canonical reference an asymmetric-backend verifier can
trust; (3) the experiment's smoke script.

Matrices are ``list[list[int|float]]``. Vectors are ``list[int|float]``.

Accumulator dtype semantics:
  - int8 inputs always accumulate in int32 in this backend (matches GPU
    tensor-core convention).
  - fp64 inputs accumulate in fp64 (no upcast available).

Output dtype: the result of ``matmul`` is downcast to ``dtype_c``. For int
output, the accumulator is wrapped to the dtype's range with two's-complement
semantics (overflow detected via the digest comparison, not silently
accepted in tolerance mode).
"""
from __future__ import annotations

import struct
import time
from typing import Any

from pkg.freivalds import prng


# --- dtype helpers --------------------------------------------------------

INT_RANGES: dict[str, tuple[int, int]] = {
    "int8": (-(1 << 7), (1 << 7) - 1),
    "int32": (-(1 << 31), (1 << 31) - 1),
}


def _is_int(dtype: str) -> bool:
    return dtype in INT_RANGES


def _wrap_int(value: int, dtype: str) -> int:
    """Two's-complement wrap of an integer to dtype range."""
    bits = 8 if dtype == "int8" else 32
    mask = (1 << bits) - 1
    v = value & mask
    if v & (1 << (bits - 1)):
        v -= 1 << bits
    return v


def _check_dtype_supported(dtype: str) -> None:
    if not prng.is_stdlib_supported(dtype):
        raise ValueError(f"stdlib backend does not support dtype {dtype!r}")


# --- backend --------------------------------------------------------------

class StdlibBackend:
    name = "stdlib"

    # -- matrix factories -------------------------------------------------

    @staticmethod
    def gen_matrix(seed: int, dtype: str, rows: int, cols: int) -> tuple[list[list], bytes]:
        _check_dtype_supported(dtype)
        return prng.gen_matrix_stdlib(seed, dtype, rows, cols)

    @staticmethod
    def read_matrix_from_bytes(buf: bytes, dtype: str, rows: int, cols: int) -> list[list]:
        _check_dtype_supported(dtype)
        return prng.read_matrix_stdlib(buf, dtype, rows, cols)

    @staticmethod
    def write_matrix_to_bytes(matrix: list[list], dtype: str) -> bytes:
        _check_dtype_supported(dtype)
        return prng.write_matrix_bytes_stdlib(matrix, dtype)

    @staticmethod
    def zeros_matrix(rows: int, cols: int, dtype: str) -> list[list]:
        _check_dtype_supported(dtype)
        if _is_int(dtype):
            return [[0] * cols for _ in range(rows)]
        return [[0.0] * cols for _ in range(rows)]

    # -- compute ----------------------------------------------------------

    @staticmethod
    def matmul(
        A: list[list],
        B: list[list],
        dtype_a: str,
        dtype_b: str,
        dtype_acc: str,
        dtype_c: str,
    ) -> list[list]:
        """``C = A @ B`` accumulated in ``dtype_acc``, downcast to ``dtype_c``."""
        _check_dtype_supported(dtype_a)
        _check_dtype_supported(dtype_b)
        _check_dtype_supported(dtype_c)
        if _is_int(dtype_acc) and not _is_int(dtype_a):
            raise ValueError(f"int accumulator with float input not supported here")

        M = len(A)
        K = len(A[0]) if M else 0
        K2 = len(B)
        N = len(B[0]) if K2 else 0
        if K != K2:
            raise ValueError(f"shape mismatch: A is {M}x{K}, B is {K2}x{N}")

        # Transpose B for cache friendliness in pure Python.
        Bt = [[B[k][n] for k in range(K)] for n in range(N)]

        if _is_int(dtype_acc):
            C = [[0] * N for _ in range(M)]
            for i in range(M):
                Ai = A[i]
                for j in range(N):
                    Btj = Bt[j]
                    s = 0
                    for k in range(K):
                        s += Ai[k] * Btj[k]
                    if _is_int(dtype_c):
                        s = _wrap_int(s, dtype_c)
                    C[i][j] = s
        else:
            C = [[0.0] * N for _ in range(M)]
            for i in range(M):
                Ai = A[i]
                for j in range(N):
                    Btj = Bt[j]
                    s = 0.0
                    for k in range(K):
                        s += Ai[k] * Btj[k]
                    C[i][j] = s
        return C

    @staticmethod
    def matvec(
        A: list[list],
        v: list,
        dtype_acc: str,
        dtype_out: str,
    ) -> list:
        """``out = A @ v``."""
        M = len(A)
        K = len(A[0]) if M else 0
        if len(v) != K:
            raise ValueError(f"shape mismatch: A is {M}x{K}, v is len {len(v)}")
        if _is_int(dtype_acc):
            out = [0] * M
            for i in range(M):
                Ai = A[i]
                s = 0
                for k in range(K):
                    s += Ai[k] * v[k]
                if _is_int(dtype_out):
                    s = _wrap_int(s, dtype_out)
                out[i] = s
        else:
            out = [0.0] * M
            for i in range(M):
                Ai = A[i]
                s = 0.0
                for k in range(K):
                    s += Ai[k] * v[k]
                out[i] = s
        return out

    # -- vectors ----------------------------------------------------------

    @staticmethod
    def random_vector(seed: int, dtype: str, n: int) -> list:
        """A deterministic vector for use as Freivalds' ``r``.

        Uses the same PRNG as :func:`gen_matrix` with ``rows=1, cols=n``.
        For integer dtypes, ``r`` is in the dtype's range; for float, it is
        in roughly ``[-1, 1)`` (the prng twiddle pins magnitudes).
        """
        canonical = prng.gen_matrix_bytes(seed, dtype, 1, n)
        return prng.read_matrix_stdlib(canonical, dtype, 1, n)[0]

    @staticmethod
    def vec_inf_norm(v: list) -> float:
        if not v:
            return 0.0
        return max(abs(float(x)) for x in v)

    @staticmethod
    def vec_max_abs_diff(u: list, v: list) -> float:
        if len(u) != len(v):
            raise ValueError(f"length mismatch: {len(u)} vs {len(v)}")
        if not u:
            return 0.0
        return max(abs(float(a) - float(b)) for a, b in zip(u, v))

    @staticmethod
    def vec_exact_equal(u: list, v: list) -> bool:
        if len(u) != len(v):
            return False
        return all(a == b for a, b in zip(u, v))

    # -- timing & device --------------------------------------------------

    @staticmethod
    def perf_time_ms() -> float:
        return time.perf_counter() * 1000.0

    @staticmethod
    def device_info() -> dict[str, Any]:
        return {"device": "cpu", "device_name": "stdlib"}
