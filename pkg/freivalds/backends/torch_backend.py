"""Torch backend for the Freivalds prover/verifier.

Same shape as :class:`StdlibBackend`, but matrices are ``torch.Tensor``
and ``matmul`` runs on the GPU when ``device='cuda'``. Produces canonical
bytes byte-identical to the stdlib backend for the dtypes both support
(int8, int32, fp64) — that is the cross-implementation invariant the
attestation protocol relies on.

GPU dtype routing (prover side, on H100/GH200):

  - ``bf16``, ``fp16``: torch.matmul on bf16/fp16 inputs uses tensor cores
    via cuBLAS. Accumulator is fp32 (matches the deterministic stack).
  - ``fp32``: torch.matmul with TF32 disabled — true fp32 tensor-core
    semantics. (TF32 enabled would change the answer subtly; we want a
    soundness-clean default.)
  - ``fp64``: torch.matmul (no tensor-core acceleration on H100; CUDA
    cores still saturate for large dims).
  - ``int8``: ``torch._int_mm`` if available (cuBLAS int8 tensor cores),
    else cast to int32 for an fp32-equivalent fallback. Output is int32.
  - ``fp8_e4m3``: ``torch._scaled_mm`` with scale=1.0 if available, else
    fp32 promotion. Output dtype declared by the spec.

Verifier-side ``matvec`` calls accumulate in int64 (for int dtypes) or fp64
(for floats) so the cheap O(n^2) check has more headroom than the prover's
matmul. This is crucial: if the verifier truncates the same way the prover
does, a wrong C may go undetected.
"""
from __future__ import annotations

from typing import Any

import torch

from pkg.freivalds import prng


_DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "int8": torch.int8,
    "int32": torch.int32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "fp8_e4m3": getattr(torch, "float8_e4m3fn", torch.float16),
}

_HAS_FP8 = hasattr(torch, "float8_e4m3fn") and hasattr(torch, "_scaled_mm")
_HAS_INT_MM = hasattr(torch, "_int_mm")


def _to_torch(buf: bytes, dtype: str, rows: int, cols: int) -> torch.Tensor:
    """Reinterpret canonical bytes as a torch tensor on the configured device.

    Uses ``frombuffer`` to keep the bytes identical to the stdlib path.
    """
    td = _DTYPE_TO_TORCH[dtype]
    flat = torch.frombuffer(bytearray(buf), dtype=td)
    return flat.view(rows, cols).clone()


def _from_torch(t: torch.Tensor, dtype: str) -> bytes:
    """Pack a torch tensor at ``dtype`` back into canonical row-major bytes."""
    td = _DTYPE_TO_TORCH[dtype]
    if t.dtype != td:
        t = t.to(td)
    return bytes(t.contiguous().cpu().view(torch.uint8).numpy().tobytes())


# --- backend --------------------------------------------------------------

class TorchBackend:
    name = "torch"

    def __init__(self, device: str = "cuda") -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; cannot use device=cuda")
        self.device = torch.device(device)
        # Soundness-clean default: TF32 off so fp32 means fp32.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # -- matrix factories -------------------------------------------------

    def gen_matrix(self, seed: int, dtype: str, rows: int, cols: int) -> tuple[torch.Tensor, bytes]:
        canonical = prng.gen_matrix_bytes(seed, dtype, rows, cols)
        t = _to_torch(canonical, dtype, rows, cols).to(self.device)
        return t, canonical

    def read_matrix_from_bytes(self, buf: bytes, dtype: str, rows: int, cols: int) -> torch.Tensor:
        return _to_torch(buf, dtype, rows, cols).to(self.device)

    def write_matrix_to_bytes(self, matrix: torch.Tensor, dtype: str) -> bytes:
        return _from_torch(matrix, dtype)

    def zeros_matrix(self, rows: int, cols: int, dtype: str) -> torch.Tensor:
        return torch.zeros(rows, cols, dtype=_DTYPE_TO_TORCH[dtype], device=self.device)

    # -- compute (prover side) -------------------------------------------

    def matmul(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        dtype_a: str,
        dtype_b: str,
        dtype_acc: str,
        dtype_c: str,
    ) -> torch.Tensor:
        td_acc = _DTYPE_TO_TORCH[dtype_acc]
        td_c = _DTYPE_TO_TORCH[dtype_c]

        if dtype_a == "int8" and dtype_b == "int8":
            if _HAS_INT_MM and self.device.type == "cuda":
                # cuBLAS int8 tensor cores; accumulator is int32 by definition.
                C = torch._int_mm(A.contiguous(), B.contiguous())
            else:
                C = (A.to(torch.int32) @ B.to(torch.int32))
            if td_c == torch.int32:
                return C
            return C.to(td_c)

        if dtype_a == "fp8_e4m3" and dtype_b == "fp8_e4m3" and _HAS_FP8 and self.device.type == "cuda":
            scale = torch.ones(1, device=self.device, dtype=torch.float32)
            # _scaled_mm requires column-major B in some torch versions;
            # transpose-then-transpose if needed. The simple case suffices
            # for square calibration.
            try:
                C = torch._scaled_mm(A, B.t().contiguous().t(), scale_a=scale, scale_b=scale, out_dtype=td_c)
                return C
            except Exception:
                # Fall through to fp32 promotion path below.
                pass

        # Float (and int32) path: cast to acc dtype, matmul, cast to c.
        if dtype_acc in ("fp16", "bf16"):
            # Tensor cores on H100 prefer the full input dtype matching dtype_acc here.
            Acc = (A.to(td_acc) @ B.to(td_acc))
        else:
            Acc = (A.to(td_acc) @ B.to(td_acc))
        if Acc.dtype != td_c:
            Acc = Acc.to(td_c)
        return Acc

    # -- compute (verifier side) -----------------------------------------

    def matvec(
        self,
        A: torch.Tensor,
        v: torch.Tensor,
        dtype_acc: str,
        dtype_out: str,
    ) -> torch.Tensor:
        # The verifier always accumulates in a wide enough dtype that no
        # honest int matmul can overflow and no honest float matmul loses
        # precision below the prover's noise. CUDA has no int64 matmul, so
        # int paths fall back to CPU — verifier matvec is O(n^2), this is
        # cheap and keeps the soundness story clean.
        if dtype_acc in ("int8", "int32"):
            return (A.to(torch.int64).cpu() @ v.to(torch.int64).cpu())
        # Floats: fp64 on GPU is supported and tight enough for honest noise.
        return (A.to(torch.float64) @ v.to(torch.float64))

    # -- vectors ----------------------------------------------------------

    def random_vector(self, seed: int, dtype: str, n: int) -> torch.Tensor:
        canonical = prng.gen_matrix_bytes(seed, dtype, 1, n)
        t = _to_torch(canonical, dtype, 1, n).to(self.device).view(n)
        return t

    @staticmethod
    def vec_inf_norm(v: torch.Tensor) -> float:
        if v.numel() == 0:
            return 0.0
        return float(v.abs().max().item())

    @staticmethod
    def vec_max_abs_diff(u: torch.Tensor, v: torch.Tensor) -> float:
        if u.numel() == 0:
            return 0.0
        return float((u.to(torch.float64) - v.to(torch.float64)).abs().max().item())

    @staticmethod
    def vec_exact_equal(u: torch.Tensor, v: torch.Tensor) -> bool:
        if u.shape != v.shape:
            return False
        return bool(torch.equal(u.to(torch.int64), v.to(torch.int64)))

    # -- timing & device --------------------------------------------------

    def perf_time_ms(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        import time
        return time.perf_counter() * 1000.0

    def device_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"device": str(self.device), "device_name": ""}
        if self.device.type == "cuda":
            idx = self.device.index or 0
            info["device_name"] = torch.cuda.get_device_name(idx)
            try:
                info["nvml_clock_mhz"] = int(torch.cuda.clock_rate(idx) // 1000) if hasattr(torch.cuda, "clock_rate") else None
            except Exception:
                info["nvml_clock_mhz"] = None
            try:
                info["nvml_temp_c"] = None  # filled in by external NVML sampler
            except Exception:
                pass
        return info
