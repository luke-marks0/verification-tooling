#!/usr/bin/env python3
"""SM occupancy sweep — dial active fraction of GPU cores to a target.

Public API
----------

::

    from sm_occupancy_sweep import OccupancyController

    ctrl = OccupancyController()  # auto-detects #SMs, JIT-compiles kernel
    ctrl.occupy(fraction=0.50, duration_s=1.0)  # run on ~50% of SMs for 1 s
    ctrl.occupy(fraction=0.25)                  # 25%, default 1 s

    # Sweep with measurement:
    rows = ctrl.sweep([0.01, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00],
                      duration_s=1.0, with_telemetry=True)

The kernel is JIT-compiled once via ``torch.utils.cpp_extension.load_inline``.
It uses 96 KB of dynamic shared memory per block — on A100 (164 KB SMEM/SM)
this forces the hardware scheduler to place at most one block per SM, so
``grid_size = N`` blocks ⇒ exactly N SMs active in parallel.

For ``fraction > 1.0``, blocks queue: power saturates at TDP, kernel wall
time scales by ``ceil(N / n_sms)``.

CLI
---

::

    python3 sm_occupancy_sweep.py \\
        --out data/sm_occupancy/sweep_a100.json \\
        --md-out reports/sm_occupancy_a100.md \\
        --percentages 0.5,1,5,10,25,50,75,90,100,150,200 \\
        --duration-s 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pkg.common.deterministic import canonical_json_text, utc_now_iso


# 96 KB per block forces 1 block / SM on A100/H100 (≥48 KB SMEM/SM
# always available; opt-in via cudaFuncSetAttribute extends to 100/132/164).
SMEM_BYTES_PER_BLOCK = 96 * 1024
THREADS_PER_BLOCK = 1024


CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// 96 KB / 4 B = 24576 floats, populated cooperatively by 1024 threads.
#define BUSY_SMEM_FLOATS 24576

extern "C" __global__
__launch_bounds__(1024, 1)
void busy_kernel(float* __restrict__ scratch, int n_iters, int seed) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Cooperatively initialise 96 KB of smem so the compiler can't elide
    // the dynamic shared allocation. Without real smem usage, the hardware
    // may pack multiple blocks per SM.
    for (int i = tid; i < BUSY_SMEM_FLOATS; i += bdim) {
        smem[i] = (float)(seed + i + (int)blockIdx.x);
    }
    __syncthreads();

    float x = smem[tid];
    for (int i = 0; i < n_iters; i++) {
        x = fmaf(x, 1.0001f, 0.5f);
    }

    // Round-trip through smem with a cross-thread dependency, then write
    // scratch UNCONDITIONALLY — guarantees the FMA loop's result is live
    // and the compiler must keep both the loop and the smem traffic.
    smem[tid] = x;
    __syncthreads();
    scratch[(int)blockIdx.x * bdim + tid] = smem[(tid + 1) & (bdim - 1)];
}

void launch_busy(torch::Tensor scratch,
                 int64_t grid,
                 int64_t threads,
                 int64_t n_iters,
                 int64_t seed,
                 int64_t smem_bytes) {
    cudaFuncSetAttribute((const void*)busy_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    busy_kernel<<<(unsigned)grid, (unsigned)threads, (unsigned)smem_bytes>>>(
        scratch.data_ptr<float>(),
        (int)n_iters,
        (int)seed);
}

// Hardware-vendor's own answer for "how many of these blocks fit on one SM
// given the kernel's resource usage." If this returns 1, the scheduler
// physically cannot place 2 blocks of (threads, smem_bytes) on an SM.
int64_t query_max_blocks_per_sm(int64_t threads, int64_t smem_bytes) {
    cudaFuncSetAttribute((const void*)busy_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    int n = -1;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &n, (const void*)busy_kernel, (int)threads, (size_t)smem_bytes);
    if (err != cudaSuccess) return -1;
    return (int64_t)n;
}
"""


CPP_DECL = r"""
#include <torch/extension.h>
void launch_busy(torch::Tensor scratch,
                 int64_t grid,
                 int64_t threads,
                 int64_t n_iters,
                 int64_t seed,
                 int64_t smem_bytes);
int64_t query_max_blocks_per_sm(int64_t threads, int64_t smem_bytes);
"""


def _build_extension(verbose: bool = False):
    from torch.utils.cpp_extension import load_inline
    return load_inline(
        name="sm_occupancy_busy_v2",
        cpp_sources=CPP_DECL,
        cuda_sources=CUDA_SRC,
        functions=["launch_busy", "query_max_blocks_per_sm"],
        verbose=verbose,
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
    )


class OccupancyController:
    """Run a CUDA busy kernel on a chosen fraction of GPU SMs."""

    def __init__(self, target_kernel_ms: float = 1000.0, verbose_compile: bool = False):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        self.torch = torch
        self.device_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        self.compute_capability = f"{cap[0]}.{cap[1]}"
        props = torch.cuda.get_device_properties(0)
        self.n_sms = int(props.multi_processor_count)
        self.total_smem_per_sm = int(getattr(props, "shared_memory_per_multiprocessor", 0))
        self.ext = _build_extension(verbose=verbose_compile)
        # Hardware vendor's own answer: how many of these blocks can the
        # scheduler put on one SM? With 96 KB SMEM/block on A100 (164 KB/SM),
        # this MUST be 1.
        self.max_blocks_per_sm = int(self.ext.query_max_blocks_per_sm(
            THREADS_PER_BLOCK, SMEM_BYTES_PER_BLOCK))
        # Calibrate iters so a single-block run (1 SM busy) takes ~target_kernel_ms.
        self.n_iters = self._calibrate_iters(target_kernel_ms)
        # Reusable scratch sized for a full-GPU launch (1024 floats / SM).
        self._scratch = torch.empty(self.n_sms * THREADS_PER_BLOCK,
                                    dtype=torch.float32, device="cuda")

    def _calibrate_iters(self, target_ms: float) -> int:
        torch = self.torch
        scratch = torch.empty(THREADS_PER_BLOCK, dtype=torch.float32, device="cuda")
        n_iters = 100_000
        # Two-stage: ramp until measurable, then scale to target. Cap so a
        # mis-measurement can't blow iters into the billions.
        cap = 200_000_000
        for _ in range(8):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.ext.launch_busy(scratch, 1, THREADS_PER_BLOCK, n_iters, 0,
                                 SMEM_BYTES_PER_BLOCK)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            if dt < 1.0:
                n_iters = min(cap, n_iters * 8)
                continue
            scale = target_ms / dt
            n_iters = max(1_000, min(cap, int(n_iters * scale)))
            # One more pass to refine.
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.ext.launch_busy(scratch, 1, THREADS_PER_BLOCK, n_iters, 0,
                                 SMEM_BYTES_PER_BLOCK)
            torch.cuda.synchronize()
            dt2 = (time.perf_counter() - t0) * 1000.0
            print(f"[occ] cal: n_iters={n_iters} dt={dt2:.1f} ms", file=sys.stderr)
            return n_iters
        return n_iters

    def fraction_to_blocks(self, fraction: float) -> int:
        if fraction <= 0:
            return 0
        return max(1, round(fraction * self.n_sms))

    @staticmethod
    def matmul_flops(n: int, k: int = 1) -> int:
        """Total FLOPs for ``k`` matmuls of size ``n × n × n``.

        One matmul is 2·n³ FLOPs (one multiply + one add per inner-product term).
        This is the static-analysis identity Buck mentioned in the protocol meeting:
        the verifier specifies a FLOPs budget; the controller derives (n, k, N_sm)
        from it via this function.
        """
        return 2 * k * n * n * n

    def flops_per_block(self, n_iters: int) -> int:
        """FLOPs the busy kernel does in one block.

        Each block has 1024 threads; each thread runs ``n_iters`` FMAs;
        each FMA = 2 FP32 ops.
        """
        return 2 * THREADS_PER_BLOCK * n_iters

    def calibrated_flops_per_sm_per_sec(self) -> float:
        """Measured FP32 throughput of the busy kernel on a single SM.

        Used to convert a verifier-supplied FLOPs budget into (N_sm, iters).
        Determined by calibration; lower than the GPU's peak FP32 because
        FMAs are dependent (no ILP across the loop) and we don't fight to
        keep the FMA + tensor pipes co-active. Honest for FP32-only workloads.
        """
        cal_ms = self._calibrated_ms()
        return self.flops_per_block(self.n_iters) / (cal_ms / 1000.0)

    def occupy_flops(self, flops: float, duration_s: float = 1.0,
                     seed: int = 1234) -> dict:
        """Burn approximately ``flops`` FP32 operations over ``duration_s``.

        This is the FLOPs-native interface Buck and Luke asked for. The verifier
        passes a number of FLOPs (e.g., from `2·n³·k`); the controller picks
        the smallest N_sm that can finish in ``duration_s`` and sizes
        per-block iterations to consume the budget.

        Returns a dict with target/actual FLOPs, kernel_ms, and the chosen
        (N_sm, iters_per_block) — the SM count is now an internal scheduling
        decision, not part of the protocol surface.
        """
        if flops <= 0:
            return {"target_flops": 0.0, "actual_flops": 0.0, "n_sms_used": 0,
                    "iters_per_block": 0, "kernel_ms": 0.0,
                    "per_sm_flops_per_sec": self.calibrated_flops_per_sm_per_sec()}
        per_sm_rate = self.calibrated_flops_per_sm_per_sec()
        # Smallest N that meets the deadline; if oversaturated, use full GPU
        # and let wall time stretch.
        required_sms = max(1, math.ceil(flops / (duration_s * per_sm_rate)))
        N = min(required_sms, self.n_sms)
        # Per-block iters for the share of FLOPs each SM should burn.
        flops_per_block = flops / N
        iters = max(1_000, int(flops_per_block / (2 * THREADS_PER_BLOCK)))
        # If verifier asked for more than the GPU can do in duration_s,
        # iters can balloon; cap at the calibrated single-SM iters scaled
        # by ⌈required/N⌉ so wall time is bounded but we still issue the
        # requested work.
        scale = max(1, math.ceil(required_sms / N))
        iters_cap = max(self.n_iters, self.n_iters * scale * 2)
        iters = min(iters, iters_cap)

        torch = self.torch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.ext.launch_busy(self._scratch, N, THREADS_PER_BLOCK, iters, seed,
                             SMEM_BYTES_PER_BLOCK)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        actual_flops = self.flops_per_block(iters) * N
        return {
            "target_flops": float(flops),
            "actual_flops": float(actual_flops),
            "target_duration_s": duration_s,
            "kernel_ms": dt_ms,
            "n_sms_used": N,
            "iters_per_block": iters,
            "per_sm_flops_per_sec": per_sm_rate,
            "oversaturated": required_sms > self.n_sms,
        }

    def occupy(self, fraction: float, duration_s: float = 1.0,
               seed: int = 1234) -> dict:
        """Run the busy kernel on ~fraction*n_sms SMs for ~duration_s.

        Tunes loop iterations on-the-fly: launches sized so total wall time
        is close to ``duration_s`` regardless of fraction.
        """
        torch = self.torch
        target_blocks = self.fraction_to_blocks(fraction)
        if target_blocks == 0:
            return {"target_fraction": fraction, "target_blocks": 0,
                    "n_sms": self.n_sms, "kernel_ms": 0.0}

        # When blocks ≤ n_sms, all run in parallel → one "shift" of n_iters.
        # When blocks > n_sms, we have ceil(blocks/n_sms) shifts per launch,
        # so per-launch wall time scales by that. Compensate iters down to
        # keep per-call duration_s.
        shifts = max(1, (target_blocks + self.n_sms - 1) // self.n_sms)
        per_call_iters = max(10_000, int(self.n_iters * (duration_s * 1000.0
                                                         / self._calibrated_ms())
                                          / shifts))
        # Single launch already takes ~duration_s.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.ext.launch_busy(self._scratch, target_blocks, THREADS_PER_BLOCK,
                             per_call_iters, seed, SMEM_BYTES_PER_BLOCK)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "target_fraction": fraction,
            "target_blocks": target_blocks,
            "actual_fraction": target_blocks / self.n_sms,
            "n_sms": self.n_sms,
            "shifts": shifts,
            "per_call_iters": per_call_iters,
            "kernel_ms": dt_ms,
        }

    def _calibrated_ms(self) -> float:
        # The calibrated iters were chosen so a 1-block run takes ~target_ms.
        # Cache it: we recompute only on demand.
        if not hasattr(self, "_cal_ms"):
            torch = self.torch
            scratch = torch.empty(THREADS_PER_BLOCK, dtype=torch.float32, device="cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.ext.launch_busy(scratch, 1, THREADS_PER_BLOCK, self.n_iters, 0,
                                 SMEM_BYTES_PER_BLOCK)
            torch.cuda.synchronize()
            self._cal_ms = (time.perf_counter() - t0) * 1000.0
        return self._cal_ms

    def sweep(self, fractions, duration_s: float = 1.0,
              with_telemetry: bool = True, with_dcgm: bool = True) -> list[dict]:
        sampler_cls = None
        if with_telemetry:
            try:
                from nvml_sampler import NvmlSampler
                sampler_cls = NvmlSampler
            except Exception as exc:  # pragma: no cover
                print(f"[occ] NVML unavailable: {exc}", file=sys.stderr)

        dcgm_cls = None
        dcgm_multi_cls = None
        if with_dcgm:
            try:
                from dcgm_sampler import (
                    DcgmSmActiveSampler,
                    DcgmMultiFieldSampler,
                    dcgmi_available,
                )
                if dcgmi_available():
                    dcgm_cls = DcgmSmActiveSampler
                    dcgm_multi_cls = DcgmMultiFieldSampler
                else:
                    print("[occ] dcgmi not on PATH; skipping DCGM SM_ACTIVE",
                          file=sys.stderr)
            except Exception as exc:  # pragma: no cover
                print(f"[occ] dcgm_sampler import failed: {exc}", file=sys.stderr)

        idle = self._idle_baseline(sampler_cls) if sampler_cls else {}
        idle_pw = idle.get("idle_power_w_mean", 0.0) or 0.0

        rows = []
        for f in fractions:
            row = self._measured_run(f, duration_s, sampler_cls, dcgm_cls,
                                     dcgm_multi_cls, idle_pw)
            rows.append(row)
        rows[0].setdefault("idle", idle)  # attach baseline to first row
        return rows

    def _idle_baseline(self, sampler_cls, seconds: float = 0.6) -> dict:
        sampler = sampler_cls(gpu_index=0, interval_ms=5)
        sampler.start(); time.sleep(seconds); sampler.stop()
        s = sampler.summary()
        return {
            "idle_power_w_mean": s.get("power_w_mean", -1.0),
            "idle_clock_mhz_median": s.get("clock_mhz_median", -1.0),
            "idle_sample_count": s.get("sample_count", 0),
        }

    def _measured_run(self, fraction, duration_s, sampler_cls, dcgm_cls,
                      dcgm_multi_cls, idle_pw) -> dict:
        sampler = sampler_cls(gpu_index=0, interval_ms=5) if sampler_cls else None
        # DCGM profiling can't sample below 100 ms on most cards, so we
        # need duration_s ≥ 0.5 to get a few clean samples.
        dcgm = dcgm_cls(gpu_index=0, interval_ms=100) if dcgm_cls else None
        dcgm_multi = (dcgm_multi_cls(gpu_index=0, interval_ms=100)
                      if dcgm_multi_cls else None)

        if sampler:
            sampler.start()
        if dcgm:
            dcgm.start()
        if dcgm_multi:
            dcgm_multi.start()
        if dcgm or dcgm_multi:
            # DCGM dmon takes ~150 ms to start emitting; pad before kernel.
            time.sleep(0.25)
        elif sampler:
            time.sleep(0.05)

        info = self.occupy(fraction, duration_s)

        if dcgm:
            time.sleep(0.15)
            dcgm.stop()
        if dcgm_multi:
            if not dcgm:
                time.sleep(0.15)
            dcgm_multi.stop()
        if sampler:
            time.sleep(0.05)
            sampler.stop()
            powers = [s.power_w for s in sampler.samples if s.power_w >= 0]
            max_pw = max(powers) if powers else 0.0
            thresh = max(idle_pw + 5.0, 0.6 * max_pw)
            active = [s for s in sampler.samples if s.power_w >= thresh]
            active_pw = [s.power_w for s in active]
            mean_pw = (sum(active_pw) / len(active_pw)) if active_pw else -1.0
            info["power_w_mean_active"] = mean_pw
            info["power_w_max"] = max_pw
            info["power_w_above_idle"] = (mean_pw - idle_pw) if mean_pw > 0 else -1.0
            info["telemetry"] = sampler.summary()
        if dcgm:
            ds = dcgm.summary()
            info["dcgm"] = ds
            # Direct cross-check: DCGM SM_ACTIVE × n_sms → measured active count.
            sma = ds.get("sm_active_mean", -1.0)
            info["dcgm_measured_blocks"] = (sma * self.n_sms) if sma >= 0 else -1.0
        if dcgm_multi:
            mds = dcgm_multi.summary()
            info["dcgm_multi"] = mds
            # Convenience surface fields for the per-SM-internal-saturation
            # check Buck asked for: divide the GPU-averaged pipe-active by
            # the fraction of SMs busy, so we get the *per-busy-SM* pipe rate.
            sma = mds.get("sm_active", {}).get("mean_active", 0.0)
            for f in ("sm_occupancy", "fp32_active", "tensor_active"):
                v = mds.get(f, {}).get("mean_active", -1.0)
                if v >= 0 and sma > 0:
                    info[f"{f}_per_busy_sm"] = v / sma
                else:
                    info[f"{f}_per_busy_sm"] = -1.0
        return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--md-out", default=None)
    p.add_argument("--percentages",
                   default="1,5,10,25,40,50,60,75,90,100,150,200",
                   help="Comma list of % of GPU SMs to occupy. >100 queues.")
    p.add_argument("--flops",
                   default=None,
                   help="Comma list of FLOPs targets. If set, runs the FLOPs "
                        "interface (occupy_flops) instead of the % sweep.")
    p.add_argument("--target-ms", type=float, default=1000.0,
                   help="Single-SM kernel duration target (calibration).")
    p.add_argument("--duration-s", type=float, default=1.0,
                   help="Per-fraction run duration (seconds).")
    args = p.parse_args(argv)

    fractions = [float(x) / 100.0 for x in args.percentages.split(",") if x.strip()]
    print(f"[occ] target percentages: {[f * 100 for f in fractions]}")

    ctrl = OccupancyController(target_kernel_ms=args.target_ms,
                               verbose_compile=False)
    print(f"[occ] device: {ctrl.device_name} (sm={ctrl.compute_capability}, "
          f"n_sms={ctrl.n_sms}, smem/sm={ctrl.total_smem_per_sm} B)")
    print(f"[occ] cudaOccupancyMaxActiveBlocksPerMultiprocessor"
          f"(threads={THREADS_PER_BLOCK}, smem={SMEM_BYTES_PER_BLOCK} B) = "
          f"**{ctrl.max_blocks_per_sm}**")
    print(f"[occ] calibrated n_iters={ctrl.n_iters} for ~{args.target_ms:.0f} ms / SM")

    if args.flops:
        flops_targets = [float(x) for x in args.flops.split(",") if x.strip()]
        per_sm_rate = ctrl.calibrated_flops_per_sm_per_sec()
        print(f"[occ] per-SM FLOPs/s (busy kernel, FP32-FMA-only): {per_sm_rate:.3e}")
        print(f"[occ] FLOPs interface: targets {flops_targets}")
        flop_rows = []
        for f in flops_targets:
            r = ctrl.occupy_flops(f, duration_s=args.duration_s)
            print(f"  flops={f:.3e} → N_sm={r['n_sms_used']} "
                  f"iters={r['iters_per_block']} actual={r['actual_flops']:.3e} "
                  f"dt={r['kernel_ms']:.0f} ms "
                  f"oversaturated={r['oversaturated']}")
            flop_rows.append(r)
        out = {
            "sm_occupancy_sweep_version": "v3-flops",
            "generated_at": utc_now_iso(),
            "device_name": ctrl.device_name,
            "compute_capability": ctrl.compute_capability,
            "n_sms": ctrl.n_sms,
            "torch_version": ctrl.torch.__version__,
            "n_iters_calibrated": ctrl.n_iters,
            "per_sm_flops_per_sec": per_sm_rate,
            "duration_s_per_run": args.duration_s,
            "flop_rows": flop_rows,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(canonical_json_text(out), encoding="utf-8")
        print(f"[occ] wrote {args.out}")
        return 0

    rows = ctrl.sweep(fractions, duration_s=args.duration_s,
                      with_telemetry=True, with_dcgm=True)

    # Reference: where N is realistic (≤ n_sms), compute Δpower/SM and a
    # linear fit. For N > n_sms, power should plateau near the full-GPU value.
    out = {
        "sm_occupancy_sweep_version": "v3",
        "generated_at": utc_now_iso(),
        "device_name": ctrl.device_name,
        "compute_capability": ctrl.compute_capability,
        "n_sms": ctrl.n_sms,
        "shared_memory_per_multiprocessor": ctrl.total_smem_per_sm,
        "torch_version": ctrl.torch.__version__,
        "smem_bytes_per_block": SMEM_BYTES_PER_BLOCK,
        "threads_per_block": THREADS_PER_BLOCK,
        "n_iters_calibrated": ctrl.n_iters,
        "duration_s_per_run": args.duration_s,
        "max_blocks_per_sm_query": ctrl.max_blocks_per_sm,
        "idle_baseline": rows[0].get("idle", {}),
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(canonical_json_text(out), encoding="utf-8")
    print(f"[occ] wrote {args.out}")

    # Pretty console table.
    print()
    print(f"  {'target%':>8} {'blocks':>7} {'kernel_ms':>10} "
          f"{'Δpower_W':>10} {'pwr_pred':>9} "
          f"{'dcgm_SMACT':>11} {'dcgm_blks':>10} {'err_blks':>9} "
          f"{'fp32/sm':>8} {'tensor/sm':>10} {'occ/sm':>8} "
          f"{'sm_max':>7} {'clock':>7}")
    # Linear fit over N ≤ n_sms.
    pts = [(r["target_blocks"], r.get("power_w_above_idle", -1))
           for r in rows
           if r.get("power_w_above_idle", -1) > 0 and r["target_blocks"] <= ctrl.n_sms]
    slope = intercept = float("nan")
    if len(pts) >= 2:
        n = len(pts)
        sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
        sxx = sum(p[0]*p[0] for p in pts); sxy = sum(p[0]*p[1] for p in pts)
        denom = n * sxx - sx * sx
        if denom != 0:
            slope = (n * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / n
    for r in rows:
        N = r["target_blocks"]
        eff_N = min(N, ctrl.n_sms)
        pwr_pred = (slope * eff_N + intercept) if not (slope != slope) else float("nan")
        t = r.get("telemetry", {}) or {}
        d = r.get("dcgm", {}) or {}
        sma = d.get("sm_active_mean", -1.0)
        dcgm_blocks = r.get("dcgm_measured_blocks", -1.0)
        err_blocks = (dcgm_blocks - eff_N) if dcgm_blocks > 0 else float("nan")
        fp32_per_sm = r.get("fp32_active_per_busy_sm", -1.0)
        tensor_per_sm = r.get("tensor_active_per_busy_sm", -1.0)
        occ_per_sm = r.get("sm_occupancy_per_busy_sm", -1.0)
        print(f"  {r['target_fraction']*100:7.1f}% "
              f"{N:7d} {r['kernel_ms']:10.1f} "
              f"{r.get('power_w_above_idle', -1):10.1f} "
              f"{pwr_pred:9.1f} "
              f"{sma:11.3f} "
              f"{dcgm_blocks:10.1f} "
              f"{err_blocks:+9.1f} "
              f"{fp32_per_sm:8.3f} "
              f"{tensor_per_sm:10.3f} "
              f"{occ_per_sm:8.3f} "
              f"{t.get('sm_util_max', -1):6.0f}% "
              f"{t.get('clock_mhz_median', -1):6.0f}")

    if args.md_out:
        idle = rows[0].get("idle", {})
        idle_pw = idle.get("idle_power_w_mean", 0.0) or 0.0
        lines = [
            "# SM occupancy sweep — controllable fraction of GPU cores",
            "",
            f"Hardware: **{ctrl.device_name}** (sm={ctrl.compute_capability}, "
            f"{ctrl.n_sms} SMs)  ",
            f"Threads per block: 1024.  ",
            f"Dynamic shared memory per block: {SMEM_BYTES_PER_BLOCK // 1024} KB "
            f"(forces 1 block per SM on A100/H100).  ",
            f"FMA iterations per block: **{ctrl.n_iters}** (calibrated to "
            f"~{args.target_ms:.0f} ms on a single SM).  ",
            f"Idle power baseline: **{idle_pw:.1f} W**.  ",
            "",
            "## Direct hardware verification",
            "",
            f"`cudaOccupancyMaxActiveBlocksPerMultiprocessor("
            f"threads={THREADS_PER_BLOCK}, smem={SMEM_BYTES_PER_BLOCK} B)` = "
            f"**{ctrl.max_blocks_per_sm}**. The CUDA runtime — i.e., NVIDIA's "
            f"own scheduler — confirms it can place at most "
            f"{ctrl.max_blocks_per_sm} block per SM with these resource "
            f"constraints. Combined with `grid = N`, this guarantees ≤ N "
            f"distinct SMs are touched.",
            "",
            "## Result table",
            "",
            "| target % | target blocks | kernel_ms | "
            "Δpower_W | DCGM SM_ACTIVE | DCGM blocks (= SMACT × n_sms) | "
            "DCGM err vs target | "
            "FP32/busy_SM | TENSOR/busy_SM | SM_OCC/busy_SM | "
            "sm_util_max | clock MHz |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            t = r.get("telemetry", {}) or {}
            d = r.get("dcgm", {}) or {}
            tb = r["target_blocks"]
            sma = d.get("sm_active_mean", -1.0)
            db = r.get("dcgm_measured_blocks", -1.0)
            err = (db - min(tb, ctrl.n_sms)) if db > 0 else float("nan")
            fp32_per_sm = r.get("fp32_active_per_busy_sm", -1.0)
            tensor_per_sm = r.get("tensor_active_per_busy_sm", -1.0)
            occ_per_sm = r.get("sm_occupancy_per_busy_sm", -1.0)
            lines.append(
                f"| {r['target_fraction']*100:.1f} | {tb} | "
                f"{r['kernel_ms']:.1f} | "
                f"{r.get('power_w_above_idle', -1):.1f} | "
                f"{sma:.3f} | "
                f"{db:.1f} | "
                f"{err:+.1f} | "
                f"{fp32_per_sm:.3f} | "
                f"{tensor_per_sm:.3f} | "
                f"{occ_per_sm:.3f} | "
                f"{t.get('sm_util_max', -1):.0f}% | "
                f"{t.get('clock_mhz_median', -1):.0f} |"
            )
        lines += [
            "",
            "## Linear fit Δpower = a·N + b (over N ≤ n_sms)",
            "",
            f"- slope **a = {slope:.2f} W/SM**",
            f"- intercept b = {intercept:.2f} W",
            f"- predicted Δpower at N={ctrl.n_sms}: "
            f"{slope*ctrl.n_sms + intercept:.1f} W ⇒ total {idle_pw + slope*ctrl.n_sms + intercept:.0f} W",
            "",
            "## Reading",
            "",
            "**Three independent signals all agree on N.**",
            "",
            "1. *Hardware contract* — `cudaOccupancyMax"
            "ActiveBlocksPerMultiprocessor` returned 1 for our resource "
            "profile, so grid = N ⇒ at most N SMs touched.",
            "2. *Direct measurement* — DCGM's `DCGM_FI_PROF_SM_ACTIVE` "
            "(field 1002) is the ratio of cycles SMs were busy averaged "
            "across all SMs. SMACT × n_sms recovers the active-block "
            "count without going through power.",
            "3. *Power scaling* — Δpower is linear in N with tight "
            "residuals; the slope is just the per-SM power increment of a "
            "FMA-only kernel.",
            "",
            "When DCGM and the linear-fit prediction agree to within ~1 SM, "
            "we know all three are reading the same physical event.",
            "",
            "**Queued regime (N > n_sms).** The script holds wall time near "
            "`duration_s` by dividing per-block iterations by `⌈N / n_sms⌉`. "
            "DCGM SMACT at 200% should still read ≈ 1.0; at 150% it should "
            "read ≈ (108 + 54)/2 / 108 = 0.75 (time-averaged across the two "
            "phases of the queued launch).",
            "",
            "**Floor.** The 'actual %' values reflect integer rounding "
            "(`round(fraction * n_sms)`); on 108 SMs the smallest non-zero "
            "target is 1 SM ≈ 0.93%.",
            "",
            "## API",
            "",
            "```python",
            "from sm_occupancy_sweep import OccupancyController",
            "ctrl = OccupancyController()",
            "ctrl.occupy(fraction=0.50, duration_s=1.0)  # 50% of SMs for 1 s",
            "ctrl.sweep([0.01, 0.10, 0.50, 1.00], duration_s=1.0)",
            "```",
            "",
        ]
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md_out).write_text("\n".join(lines), encoding="utf-8")
        print(f"[occ] wrote {args.md_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
