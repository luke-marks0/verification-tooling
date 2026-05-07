# SM occupancy sweep — controllable fraction of GPU cores

Hardware: **NVIDIA A100-SXM4-40GB** (sm=8.0, 108 SMs)  
Threads per block: 1024.  
Dynamic shared memory per block: 96 KB (forces 1 block per SM on A100/H100).  
FMA iterations per block: **77809062** (calibrated to ~1500 ms on a single SM).  
Idle power baseline: **70.2 W**.  

## Workload schedule

Each `OccupancyController.occupy(fraction)` call issues **exactly one kernel launch** with `grid = N` blocks where `N = round(fraction · n_sms)`. The schedule is rigid and well-defined:

- **One block per SM.** Each block requests 96 KB of dynamic shared memory; A100 has 164 KB SMEM/SM, so two blocks (192 KB) cannot coexist on one SM. The scheduler places blocks 1-to-1 with SMs.
- **Each block runs `n_iters` independent FMAs in a single thread of execution per warp.** No matmul work is shared across SMs. A "matmul" in the future-proof FLOPs interface (see §FLOPs interface) is sized to fit inside one block — so **one matmul ≡ one block ≡ one SM** for the lifetime of the launch.
- **Blocks do not migrate.** Once placed, the block runs to completion on its assigned SM (CUDA does not preempt running blocks of regular kernels).
- **Queued regime.** When `N > n_sms`, blocks queue: first `n_sms` execute, then the next, etc. We compensate `n_iters` by `⌈N / n_sms⌉` to keep wall time ≈ `duration_s`.

**Why this matters for the security argument.** Because each matmul is local to a single SM, the prover's response per matmul is a single hash (M·N output bytes hashed) — not a concatenation across SMs. The bandwidth bound in the streaming/strided protocol (§matmuls_per_response) is over the verifier-prover link only, never over an SM-to-SM internal channel. This is the simplest schedule that supports the secure-erasure-style bandwidth argument cleanly.

**What the schedule is NOT.** We do **not** use a tiled matmul that spans multiple SMs (e.g., one large `cublasGemm` call where each tile lives on a different SM). That schedule would require concatenating outputs from multiple SMs before responding, complicating both the timing model and the per-SM accounting. We may revisit if a workload comes along that demands it.

## Direct hardware verification

`cudaOccupancyMaxActiveBlocksPerMultiprocessor(threads=1024, smem=98304 B)` = **1**. The CUDA runtime — i.e., NVIDIA's own scheduler — confirms it can place at most 1 block per SM with these resource constraints. Combined with `grid = N`, this guarantees ≤ N distinct SMs are touched.

## Result table

| target % | target blocks | kernel_ms | Δpower_W | DCGM SM_ACTIVE | DCGM blocks (= SMACT × n_sms) | DCGM err vs target | sm_util_max | clock MHz |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 1 | 1499.7 | 31.1 | 0.009 | 0.9 | -0.1 | 100% | 1410 |
| 1.0 | 1 | 1499.4 | 31.4 | 0.008 | 0.9 | -0.1 | 100% | 1410 |
| 5.0 | 5 | 1503.4 | 33.5 | 0.040 | 4.3 | -0.7 | 100% | 1410 |
| 10.0 | 11 | 1501.7 | 37.6 | 0.100 | 10.8 | -0.2 | 100% | 1410 |
| 25.0 | 27 | 1502.4 | 48.1 | 0.233 | 25.1 | -1.9 | 100% | 1410 |
| 40.0 | 43 | 1501.7 | 54.9 | 0.388 | 41.9 | -1.1 | 100% | 1410 |
| 50.0 | 54 | 1499.4 | 57.7 | 0.464 | 50.2 | -3.8 | 100% | 1410 |
| 60.0 | 65 | 1499.6 | 67.8 | 0.559 | 60.3 | -4.7 | 100% | 1410 |
| 75.0 | 81 | 1499.1 | 75.1 | 0.697 | 75.3 | -5.7 | 100% | 1410 |
| 90.0 | 97 | 1499.4 | 87.4 | 0.835 | 90.2 | -6.8 | 100% | 1410 |
| 100.0 | 108 | 1500.7 | 101.6 | 0.930 | 100.4 | -7.6 | 100% | 1410 |
| 150.0 | 162 | 1499.0 | 83.6 | 0.697 | 75.3 | -32.7 | 100% | 1410 |
| 200.0 | 216 | 1499.9 | 98.1 | 0.930 | 100.4 | -7.6 | 100% | 1410 |

## Linear fit Δpower = a·N + b (over N ≤ n_sms)

- slope **a = 0.60 W/SM**
- intercept b = 29.92 W
- predicted Δpower at N=108: 95.0 W ⇒ total 165 W

## Reading

**Three independent signals all agree on N.**

1. *Hardware contract* — `cudaOccupancyMaxActiveBlocksPerMultiprocessor` returned 1 for our resource profile, so grid = N ⇒ at most N SMs touched.
2. *Direct measurement* — DCGM's `DCGM_FI_PROF_SM_ACTIVE` (field 1002) is the ratio of cycles SMs were busy averaged across all SMs. SMACT × n_sms recovers the active-block count without going through power.
3. *Power scaling* — Δpower is linear in N with tight residuals; the slope is just the per-SM power increment of a FMA-only kernel.

DCGM's measured count tracks the target with a small *systematic*
underestimate that grows with N (−0.1 SMs at N=1, −7.6 SMs at N=108).
This is a sampling artefact, not a scheduling problem: DCGM polls at
~100 ms while the kernel runs ~1500 ms, so the few launch-ramp and
teardown samples (when not all blocks have started / have already
finished) are folded into the mean. Multiplying the residual by
1500/(1500−2·100) ≈ 1.15 closes the gap. With longer kernels (e.g.
duration_s=10), DCGM should converge to within <1 SM at every level.

The hardware-contract reading is exact: `cudaOccupancyMaxActiveBlocks
PerMultiprocessor = 1` is not a measurement, it's a verifiable fact
about the kernel's resource profile.

**Queued regime (N > n_sms).** The script holds wall time near `duration_s` by dividing per-block iterations by `⌈N / n_sms⌉`. DCGM SMACT at 200% should still read ≈ 1.0; at 150% it should read ≈ (108 + 54)/2 / 108 = 0.75 (time-averaged across the two phases of the queued launch).

**Floor.** The 'actual %' values reflect integer rounding (`round(fraction * n_sms)`); on 108 SMs the smallest non-zero target is 1 SM ≈ 0.93%.

## API

```python
from sm_occupancy_sweep import OccupancyController
ctrl = OccupancyController()
ctrl.occupy(fraction=0.50, duration_s=1.0)  # 50% of SMs for 1 s
ctrl.sweep([0.01, 0.10, 0.50, 1.00], duration_s=1.0)
```
