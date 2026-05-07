"""Compute backends for the Freivalds prover/verifier.

A backend exposes:

  - ``gen_matrix(seed, dtype, rows, cols) -> (matrix_obj, canonical_bytes)``
  - ``matmul(A, B, dtype_acc, dtype_c) -> matrix_obj``
  - ``matvec(A, v, dtype_acc, dtype_out) -> vector_obj``  (used by the verifier
     for the cheap O(n^2) Freivalds check)
  - ``read_matrix_from_bytes(buf, dtype, rows, cols) -> matrix_obj``
  - ``write_matrix_to_bytes(matrix_obj, dtype) -> bytes``
  - ``random_vector(seed, dtype, n) -> vector_obj``
  - ``zeros_matrix(rows, cols, dtype) -> matrix_obj``  (probe support)
  - ``device_info() -> dict[str, Any]``  (name, clock, temp etc., observational)

The exact ``matrix_obj`` type is backend-specific (lists for stdlib, tensors
for torch). Callers stay backend-agnostic.
"""
from __future__ import annotations

from pkg.freivalds.backends.stdlib import StdlibBackend

__all__ = ["StdlibBackend"]
