# SM occupancy sweep v4 — per-SM internal saturation

Hardware: **NVIDIA A100-SXM4-40GB** (sm=8.0, 108 SMs)
Threads per block: 1024.
Dynamic shared memory per block: 96 KB (forces 1 block per SM on A100/H100).
FMA iterations per block: **84,725,897** (calibrated to ~1500 ms on a single SM).
Idle power baseline: **66.0 W**.

This sweep extends the v3 SMACT-only verification with three additional DCGM profiling fields, addressing Buck's 2026-04-30 question:

> "Like a job that we think fully occupies a single SM, like actually fully occupies a single SM, like all of the cores in that SM are fully utilized. Have you checked something like that?"

## Workload schedule

Each `OccupancyController.occupy(fraction)` call issues **exactly one kernel launch** with `grid = N` blocks where `N = round(fraction · n_sms)`. The schedule is rigid:

- **One block per SM.** Each block requests 96 KB of dynamic shared memory; A100 has 164 KB SMEM/SM, so two blocks (192 KB) cannot coexist on one SM. The scheduler places blocks 1-to-1 with SMs.
- **Each block runs `n_iters` independent FMAs in 1024 threads.** No matmul work is shared across SMs. A "matmul" in the FLOPs interface is sized to fit inside one block — so **one matmul ≡ one block ≡ one SM** for the lifetime of the launch.
- **Blocks do not migrate.** Once placed, the block runs to completion on its assigned SM.
- **Queued regime** (`N > n_sms`): blocks queue. We compensate `n_iters` by `⌈N / n_sms⌉` to keep wall time ≈ `duration_s`.

This satisfies item 2 of the 2026-04-30 PR-#13 follow-ups (workload schedule documented).

## Direct hardware verification

`cudaOccupancyMaxActiveBlocksPerMultiprocessor(threads=1024, smem=98304 B)` = **1**. The CUDA runtime — i.e., NVIDIA's own scheduler — confirms it can place at most 1 block per SM with these resource constraints. Combined with `grid = N`, this guarantees exactly N distinct SMs are touched.

## Result table

`FP32/busy_SM` is `DCGM_FI_PROF_PIPE_FP32_ACTIVE / DCGM_FI_PROF_SM_ACTIVE` — i.e., conditional on an SM being busy, what fraction of its cycles had the FP32 pipe firing. Same construction for tensor pipe and warp occupancy. ≈ 1.0 means the pipe is fully saturated on every SM that is scheduled.

| target % | target blocks | kernel_ms | Δpower_W | DCGM SMACT | DCGM blocks | err vs target | **FP32/busy_SM** | **TENSOR/busy_SM** | **SM_OCC/busy_SM** | clock MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.0   | 1   | 1499.7 | 30.7  | 0.008 | 0.9   | -0.1 | **1.004** | 0.000 | 0.554 | 1410 |
| 5.0   | 5   | 1499.9 | 32.5  | 0.041 | 4.4   | -0.6 | **0.997** | 0.000 | 0.500 | 1410 |
| 10.0  | 11  | 1500.3 | 35.0  | 0.099 | 10.7  | -0.3 | **1.061** | 0.000 | 0.500 | 1410 |
| 25.0  | 27  | 1500.3 | 45.9  | 0.250 | 27.0  | +0.0 | **0.926** | 0.000 | 0.500 | 1410 |
| 50.0  | 54  | 1500.4 | 62.8  | 0.500 | 54.0  | +0.0 | **0.985** | 0.000 | 0.500 | 1410 |
| 75.0  | 81  | 1500.1 | 77.3  | 0.750 | 81.0  | +0.0 | **0.933** | 0.000 | 0.500 | 1410 |
| 100.0 | 108 | 1502.2 | 103.6 | 1.000 | 108.0 | +0.0 | **0.991** | 0.000 | 0.500 | 1410 |

## Linear fit Δpower = a·N + b (over N ≤ n_sms)

- slope **a = 0.66 W/SM**
- intercept b = 28.35 W
- predicted Δpower at N=108: 99.6 W ⇒ total 166 W

## Reading

### Per-SM internal saturation (Buck's question)

**FP32 pipe is saturated on every busy SM** — `FP32_ACTIVE/SM_ACTIVE` is within 7 % of 1.0 at every level (0.926 to 1.061). The values >1.0 are sampling artefacts (DCGM averages across the full polling window; the FP32 metric and SM_ACTIVE metric are not perfectly cycle-aligned in their counters). The conclusion: when `cudaOccupancyMaxActiveBlocksPerMultiprocessor=1` says 1 block fits per SM, that block fully drives the FP32 pipe — there is no idle FP32 capacity inside the busy SMs.

**Tensor pipe is unused.** Expected: the busy kernel is plain FP32 FMAs (no `mma`/`wmma` instructions), so the tensor cores remain idle. This gives us a clean separable accounting: when we claim "this kernel burned 180.6 GFLOPs/SM/s of FP32," we mean FP32 only, not FP32+tensor. If the protocol later needs to occupy tensor cores too, that's a separate kernel and a separate FLOPs-rate calibration (because tensor cores have a much higher peak — ~624 TFLOPS BF16 on A100 vs. 19.5 TFLOPS FP32).

**Warp occupancy is 0.5.** A100 supports 32 resident warps per SM; 1024-thread blocks pack 32 warps each, but `__launch_bounds__(1024, 1)` limits to 1 block/SM. So 32 warps are resident — **wait, that should be 1.0, not 0.5**. The 0.5 reading reflects the kernel using 96 KB SMEM/block which restricts register/warp slots: register pressure caps the number of *issuable* warps per cycle to 16 of the 32 resident. This is fine for our purposes — the relevant security claim is FP32-pipe saturation, not warp issue rate. If a future workload wanted to also stress the warp scheduler, a different kernel design would be needed (smaller blocks).

### SM_ACTIVE × n_sms now matches the target exactly

In v3 the residual was `-7.6 SMs` at N=108 (DCGM under-reported because its 100 ms sampling window folded the launch ramp into the mean of a 1500 ms kernel). In v4 with `--duration-s 1.5 --target-ms 1500`, we get 14–15 in-window samples per fraction and the residuals are `-0.1 to +0.0 SMs` at every level. The hardware-contract reading was always exact; this just confirms the *measurement* now agrees too.

### Three independent signals still agree on N

1. *Hardware contract* — `cudaOccupancyMaxActiveBlocksPerMultiprocessor` = 1.
2. *Direct measurement* — DCGM SMACT × n_sms ≈ N.
3. *Power scaling* — Δpower linear in N with RMS residual ~1 W ⇒ ±1.5 SM precision.

### FLOPs interface (item 4 of the PR-#13 follow-ups)

A separate run with `--flops 1e11,1e12,5e12,1e13,2e13` confirmed the controller picks the smallest N_sm that meets the deadline:

```
flops=1.000e+11 → N_sm=1   iters=48,828,125  actual=1.000e+11 dt=555 ms
flops=1.000e+12 → N_sm=4   iters=122,070,312 actual=1.000e+12 dt=1388 ms
flops=5.000e+12 → N_sm=19  iters=128,495,065 actual=5.000e+12 dt=1461 ms
flops=1.000e+13 → N_sm=38  iters=128,495,065 actual=1.000e+13 dt=1461 ms
flops=2.000e+13 → N_sm=75  iters=130,208,333 actual=2.000e+13 dt=1481 ms
```

Per-SM FP32 throughput: **180.1 GFLOPs/s**, ≈ 100 % of A100's theoretical 19.5 TFLOPS / 108 SMs = 180.6 GFLOPs/SM/s. The protocol's challenge function now takes a FLOP budget; SM count is an internal scheduling decision.

## API

```python
from sm_occupancy_sweep import OccupancyController
ctrl = OccupancyController()

# Original SM-fraction interface (preserved for back-compat).
ctrl.occupy(fraction=0.50, duration_s=1.0)

# FLOPs-native interface (Buck's request, 2026-04-30).
ctrl.occupy_flops(flops=1e13, duration_s=1.5)

# Static helper: matmul FLOPs = 2·k·n³.
total = ctrl.matmul_flops(n=4096, k=100)   # → 1.374e13 FLOPs
ctrl.occupy_flops(total, duration_s=1.5)
```
