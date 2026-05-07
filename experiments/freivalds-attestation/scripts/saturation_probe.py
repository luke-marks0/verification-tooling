#!/usr/bin/env python3
"""Saturation probe: NVML sample DURING a tight matmul loop.

The full calibration script reports per-challenge end-to-end time, which
is dominated by PRNG expansion + host/device transfer, not by the matmul.
The util numbers there read 0-1% even though the matmul itself is on
tensor cores. This script isolates the matmul:

  * generate A, B once (no per-iter PRNG / transfer)
  * start an NVML sampler at 5ms
  * run K matmuls back-to-back, sync once at the end
  * stop the sampler and report (TF/s, % peak, sm_util band, clock, power)

Output: ``data/saturation_probe_v1.json``.

Confirms the plan's saturation claim: a large square GEMM hits ≥75% of
peak tensor-core FLOP/s on H100/GH200 and the GPU is unavailable for
other work during the kernel.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pkg.common.deterministic import canonical_json_text, utc_now_iso


_PEAK_TFLOPS_BY_GPU: dict[str, dict[str, float]] = {
    # H100 SXM5 / GH200 — Hopper SM90, tensor-core peak (vendor specs).
    "H100":  {"int8": 1979.0, "bf16": 989.0, "fp16": 989.0, "fp32": 67.0, "fp64": 67.0},
    "GH200": {"int8": 1979.0, "bf16": 989.0, "fp16": 989.0, "fp32": 67.0, "fp64": 67.0},
    # A100 SXM4 — Ampere SM80.
    "A100":  {"int8": 624.0,  "bf16": 312.0, "fp16": 312.0, "fp32": 19.5, "fp64": 19.5},
    # A10 — Ampere SM86.
    "A10":   {"int8": 250.0,  "bf16": 125.0, "fp16": 125.0, "fp32": 31.0, "fp64": 0.97},
    # L4 / L40 / RTX 4090 (sm_89) — Ada.
    "L4":    {"int8": 484.0,  "bf16": 121.0, "fp16": 242.0, "fp32": 30.3, "fp64": 0.5},
    "L40":   {"int8": 724.0,  "bf16": 181.0, "fp16": 362.0, "fp32": 90.0, "fp64": 1.4},
    "4090":  {"int8": 660.0,  "bf16": 165.0, "fp16": 330.0, "fp32": 82.6, "fp64": 1.3},
}


def _peak_tflops_for(device_name: str, dtype_name: str) -> float:
    name = device_name.upper()
    # Longest key first so "L40S" wins over "L40", "RTX A6000" over "A6000",
    # "GH200" over "H200" and "H100".
    for key in sorted(_PEAK_TFLOPS_BY_GPU, key=lambda s: -len(s)):
        if key in name:
            return _PEAK_TFLOPS_BY_GPU[key].get(dtype_name, 0.0)
    return 0.0


def run_one(dim: int, dtype_name: str, iters: int, sampler_cls) -> dict:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    td = {
        "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
        "fp64": torch.float64, "int8": torch.int8,
    }[dtype_name]
    if dtype_name == "int8":
        A = torch.randint(-100, 100, (dim, dim), dtype=torch.int8, device="cuda")
        B = torch.randint(-100, 100, (dim, dim), dtype=torch.int8, device="cuda")
    else:
        A = torch.randn(dim, dim, dtype=td, device="cuda")
        B = torch.randn(dim, dim, dtype=td, device="cuda")

    # Warmup
    for _ in range(5):
        if dtype_name == "int8":
            _ = torch._int_mm(A.contiguous(), B.contiguous())
        else:
            _ = A @ B
    torch.cuda.synchronize()

    sampler = sampler_cls(gpu_index=0, interval_ms=5) if sampler_cls else None
    if sampler:
        sampler.start()

    t0 = time.perf_counter()
    for _ in range(iters):
        if dtype_name == "int8":
            C = torch._int_mm(A.contiguous(), B.contiguous())
        else:
            C = A @ B
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    if sampler:
        sampler.stop()

    flops = 2 * dim ** 3 * iters
    tflops = flops / dt / 1e12
    peak = _peak_tflops_for(torch.cuda.get_device_name(0), dtype_name)
    out = {
        "dim": dim, "dtype": dtype_name, "iters": iters,
        "wall_time_ms": dt * 1000.0,
        "per_iter_ms": dt * 1000.0 / iters,
        "tflops_observed": tflops,
        "peak_tflops": peak,
        "fraction_of_peak": tflops / peak if peak else 0.0,
    }
    if sampler:
        s = sampler.summary()
        out["telemetry"] = s
        # Number of distinct samples that saw >50% util — a coarse "GPU was
        # busy with matmul" check.
        busy = sum(1 for sm in sampler.samples if sm.sm_util >= 50)
        out["telemetry"]["samples_above_50pct"] = busy
        out["telemetry"]["samples_below_5pct"] = sum(1 for sm in sampler.samples if 0 <= sm.sm_util < 5)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--md-out", default=None)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args(argv)

    sampler_cls = None
    try:
        from nvml_sampler import NvmlSampler
        sampler_cls = NvmlSampler
    except Exception as exc:
        print(f"[sat] NVML unavailable ({exc}); continuing without telemetry",
              file=sys.stderr)

    cells = []
    cases = [
        (4096, "int8"), (8192, "int8"),
        (4096, "bf16"), (8192, "bf16"),
        (4096, "fp16"), (8192, "fp16"),
        (4096, "fp32"), (8192, "fp32"),
    ]
    import torch
    print(f"[sat] device: {torch.cuda.get_device_name(0)}, iters per case: {args.iters}")
    for dim, dt in cases:
        try:
            row = run_one(dim, dt, args.iters, sampler_cls)
        except Exception as exc:
            print(f"  SKIP {dt} {dim}: {exc}", file=sys.stderr)
            continue
        t = row.get("telemetry", {})
        print(f"  {dt:5} dim={dim:5}  per_iter={row['per_iter_ms']:.2f}ms  "
              f"tflops={row['tflops_observed']:.1f}  ({row['fraction_of_peak']:.0%} peak)  "
              f"sm_util_med={t.get('sm_util_median', -1):.0f}%  "
              f"sm_util_max={t.get('sm_util_max', -1):.0f}%  "
              f"clock={t.get('clock_mhz_median', -1):.0f}MHz  "
              f"power={t.get('power_w_mean', -1):.0f}W")
        cells.append(row)

    out = {
        "saturation_probe_version": "v1",
        "generated_at": utc_now_iso(),
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cells": cells,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(canonical_json_text(out), encoding="utf-8")
    print(f"[sat] wrote {args.out}")

    if args.md_out:
        lines = [
            "# Saturation probe",
            "",
            f"Hardware: {out['device_name']}  ",
            f"Tight matmul loop ({args.iters} iters per cell), NVML sampled at 5ms during the loop.",
            f"Generated: {out['generated_at']}",
            "",
            "| dtype | dim | per_iter_ms | TF/s | % peak | sm_util median | sm_util max | clock MHz | power W |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in cells:
            t = r.get("telemetry", {})
            lines.append(
                f"| {r['dtype']} | {r['dim']} | {r['per_iter_ms']:.2f} | "
                f"{r['tflops_observed']:.1f} | {r['fraction_of_peak']:.0%} | "
                f"{t.get('sm_util_median', -1):.0f}% | {t.get('sm_util_max', -1):.0f}% | "
                f"{t.get('clock_mhz_median', -1):.0f} | {t.get('power_w_mean', -1):.0f} |"
            )
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[sat] wrote {args.md_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
