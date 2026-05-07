#!/usr/bin/env python3
"""Calibration sweep on a real GPU.

For each ``(dim, dtype)`` cell in the sweep:
  * build a single-matmul Challenge with M=K=N=dim;
  * warmup (1 run, untimed);
  * for ``--trials`` honest trials: time the matmul, run Freivalds,
    record max_abs_diff, NVML samples (sm_util/clock/power/temp);
  * aggregate per-cell stats (median time, IQR, observed TFLOPS,
    util band, suggested ε).

Output: ``data/calibration_v1.json`` with a row per (dim, dtype) and a
top-level ``hardware`` block. Designed to run on a single H100/GH200.

Run:
  python3 experiments/freivalds-attestation/scripts/calibrate.py \\
    --out experiments/freivalds-attestation/data/calibration_v1.json \\
    --trials 20
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

from pkg.common.deterministic import canonical_json_text, utc_now_iso
from pkg.freivalds import (
    Challenge,
    ComparisonMode,
    MatmulSpec,
    Tolerance,
    execute_challenge,
    verify_response,
)
from pkg.freivalds.spec import INTEGER_DTYPES

# Imported here so a CPU-only `--help` still works
def _import_torch_backend():
    from pkg.freivalds.backends.torch_backend import TorchBackend
    return TorchBackend


# Default sweep — dims at successive doublings; small dims are non-saturating
# but useful for the dtype-coverage probes.
DEFAULT_SQUARE_DIMS = (1024, 2048, 4096, 8192)

# Each dtype tuple is (dtype_a, dtype_b, dtype_acc, dtype_c, comparison).
DEFAULT_DTYPE_PROFILES = (
    ("int8",  "int8",  "int32", "int32", "bitwise"),
    ("bf16",  "bf16",  "fp32",  "bf16",  "tolerance"),
    ("fp16",  "fp16",  "fp32",  "fp16",  "tolerance"),
    ("fp32",  "fp32",  "fp32",  "fp32",  "tolerance"),
)


def _percentiles(xs, ps):
    if not xs:
        return [0.0] * len(ps)
    s = sorted(xs)
    out = []
    for p in ps:
        idx = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
        out.append(float(s[idx]))
    return out


def _peak_tflops(dtype: str) -> float:
    """Approximate peak tensor-core TF/s on H100 SXM5 / GH200 (rough; for ratio reporting)."""
    return {
        "int8": 1979.0,
        "fp8_e4m3": 1979.0,
        "bf16": 989.0,
        "fp16": 989.0,
        "fp32": 67.0,    # FP32 cuda cores; tensor-core fp32 TF32 would be 989
        "fp64": 67.0,
    }.get(dtype, 0.0)


def _run_cell(backend, dim: int, profile, trials: int, sampler_cls):
    dtype_a, dtype_b, dtype_acc, dtype_c, comp = profile
    is_int = dtype_a in INTEGER_DTYPES
    tol = None if comp == "bitwise" else Tolerance(atol=1e-2, rtol=1e-2)

    # Cell uses a single matmul; we issue ``trials`` independent challenges
    # so each trial gets a fresh seed (cache attacks would lose).
    timings = []
    diffs = []
    util_meds = []
    clock_meds = []
    power_means = []
    temps = []
    backend_was_passed = []

    for trial in range(trials + 1):  # first is warmup
        seed_a = 1_000_000 + trial * 31 + dim
        seed_b = 2_000_000 + trial * 37 + dim
        spec = MatmulSpec(
            id=f"cell-{dtype_a}-{dim}",
            M=dim, K=dim, N=dim,
            dtype_a=dtype_a, dtype_b=dtype_b, dtype_acc=dtype_acc, dtype_c=dtype_c,
            seed_a=seed_a, seed_b=seed_b,
            comparison=ComparisonMode(comp),
            tolerance=tol,
        )
        ch = Challenge(challenge_id=f"calib.{dtype_a}.{dim}.{trial}", matmuls=(spec,))

        if sampler_cls is not None and trial > 0:
            sampler = sampler_cls(gpu_index=0, interval_ms=10)
            sampler.start()
        else:
            sampler = None

        t0 = time.perf_counter()
        resp = execute_challenge(ch, backend)
        t1 = time.perf_counter()

        if sampler is not None:
            sampler.stop()

        # Verify locally; we want max_abs_diff per trial.
        report = verify_response(ch, resp, backend, r_seed_source=lambda: 1234567 + trial)
        v = report.matmuls[0]

        if trial == 0:
            continue  # warmup

        # `resp.results[0].wall_time_ms` is the matmul-only time the prover
        # reported (cuda-synced). The wall-clock around execute_challenge
        # includes PRNG expansion, host->device upload, device->host download,
        # and base64 — useful for end-to-end perf but not for tflops math.
        matmul_ms = resp.results[0].wall_time_ms
        timings.append(matmul_ms)
        diffs.append(v.max_abs_diff)
        backend_was_passed.append(v.passed)

        if sampler is not None:
            s = sampler.summary()
            util_meds.append(s.get("sm_util_median", -1))
            clock_meds.append(s.get("clock_mhz_median", -1))
            power_means.append(s.get("power_w_mean", -1))
            temps.append(s.get("temp_c_max", -1))

    p25, p50, p75, p99 = _percentiles(timings, (25, 50, 75, 99))
    flops = 2 * dim * dim * dim
    tflops_obs = (flops / (p50 / 1000.0)) / 1e12 if p50 > 0 else 0.0
    peak = _peak_tflops(dtype_a)
    fraction_of_peak = tflops_obs / peak if peak > 0 else 0.0

    # Suggested ε is the 99th percentile of observed honest max_abs_diff,
    # with a 2x safety margin. Bitwise cells don't need this.
    diff_p50, diff_p99 = _percentiles(diffs, (50, 99))
    suggested_atol = 0.0 if is_int else max(diff_p99 * 2.0, 1e-12)
    suggested_rtol = 0.0 if is_int else 1e-2

    return {
        "dim": dim,
        "dtype_a": dtype_a, "dtype_b": dtype_b, "dtype_acc": dtype_acc, "dtype_c": dtype_c,
        "comparison": comp,
        "trials": trials,
        "wall_time_ms": {
            "p25": p25, "p50": p50, "p75": p75, "p99": p99,
            "iqr_ratio": (p75 - p25) / p50 if p50 > 0 else 0.0,
        },
        "honest_diff": {"p50": diff_p50, "p99": diff_p99},
        "honest_pass_rate": sum(1 for x in backend_was_passed if x) / max(1, len(backend_was_passed)),
        "throughput": {
            "observed_tflops": tflops_obs,
            "peak_tflops": peak,
            "fraction_of_peak": fraction_of_peak,
        },
        "telemetry": {
            "sm_util_median_p50": _percentiles(util_meds, (50,))[0],
            "clock_mhz_median_p50": _percentiles(clock_meds, (50,))[0],
            "power_w_mean_p50": _percentiles(power_means, (50,))[0],
            "temp_c_max": float(max(temps)) if temps else -1.0,
        },
        "suggested_tolerance": {"atol": suggested_atol, "rtol": suggested_rtol},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="Path to write calibration_v1.json")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--dims", type=int, nargs="*", default=list(DEFAULT_SQUARE_DIMS))
    p.add_argument("--dtypes", nargs="*", default=None,
                   help="Subset of dtype_a values (default: all)")
    p.add_argument("--no-monitor", action="store_true",
                   help="Skip NVML sampler (useful when libnvidia-ml is unavailable)")
    args = p.parse_args(argv)

    TorchBackend = _import_torch_backend()
    backend = TorchBackend(device="cuda")

    sampler_cls = None
    if not args.no_monitor:
        try:
            from experiments_freivalds_nvml import NvmlSampler  # type: ignore
            sampler_cls = NvmlSampler
        except Exception:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            try:
                from nvml_sampler import NvmlSampler
                sampler_cls = NvmlSampler
            except Exception as exc:
                print(f"[calibrate] NVML unavailable ({exc}); continuing without monitoring",
                      file=sys.stderr)

    profiles = DEFAULT_DTYPE_PROFILES
    if args.dtypes:
        wanted = set(args.dtypes)
        profiles = tuple(prof for prof in profiles if prof[0] in wanted)
        if not profiles:
            print(f"[calibrate] no profiles matched {args.dtypes!r}", file=sys.stderr)
            return 2

    import torch
    hw = {
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
    }

    cells = []
    print(f"[calibrate] device: {hw['device_name']} (cc={hw['device_capability']}, torch={hw['torch_version']})")
    for prof in profiles:
        for dim in args.dims:
            print(f"[calibrate] {prof[0]} dim={dim} trials={args.trials} ...", flush=True)
            try:
                row = _run_cell(backend, dim, prof, args.trials, sampler_cls)
            except Exception as exc:
                print(f"  SKIP {prof[0]} {dim}: {exc}", file=sys.stderr)
                continue
            wt = row["wall_time_ms"]
            tp = row["throughput"]
            print(
                f"  median={wt['p50']:.2f} ms  iqr={wt['iqr_ratio']:.1%}  "
                f"tflops={tp['observed_tflops']:.1f} ({tp['fraction_of_peak']:.0%} peak)  "
                f"util={row['telemetry']['sm_util_median_p50']:.0f}%  "
                f"diff_p99={row['honest_diff']['p99']:.3g}",
                flush=True,
            )
            cells.append(row)

    out = {
        "calibration_version": "v1",
        "generated_at": utc_now_iso(),
        "hardware": hw,
        "cells": cells,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(canonical_json_text(out), encoding="utf-8")
    print(f"[calibrate] wrote {args.out}  ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
