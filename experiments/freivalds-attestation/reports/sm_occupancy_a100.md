# SM occupancy sweep — controllable fraction of GPU cores

Hardware: **NVIDIA A100-SXM4-40GB** (sm=8.0, 108 SMs)  
Threads per block: 1024.  
Dynamic shared memory per block: 96 KB (forces 1 block per SM on A100/H100).  
FMA iterations per block: **55889401** (calibrated to ~1000 ms on a single SM).  
Idle power baseline: **92.3 W**.  

## Result table

| target % | target blocks | actual % | kernel_ms | power_W (active) | Δpower_W | Δpower / target_block | sm_util_max | clock MHz |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 1 | 0.9 | 999.9 | 98.6 | 6.4 | 6.35 | 100% | 1410 |
| 1.0 | 1 | 0.9 | 999.9 | 98.6 | 6.3 | 6.31 | 100% | 1410 |
| 5.0 | 5 | 4.6 | 1005.0 | 101.3 | 9.1 | 1.82 | 100% | 1410 |
| 10.0 | 11 | 10.2 | 1005.5 | 105.8 | 13.6 | 1.24 | 100% | 1410 |
| 25.0 | 27 | 25.0 | 1006.5 | 117.5 | 25.2 | 0.93 | 100% | 1410 |
| 40.0 | 43 | 39.8 | 1006.4 | 127.7 | 35.4 | 0.82 | 100% | 1410 |
| 50.0 | 54 | 50.0 | 1007.9 | 133.9 | 41.6 | 0.77 | 100% | 1410 |
| 60.0 | 65 | 60.2 | 1004.5 | 142.2 | 49.9 | 0.77 | 100% | 1410 |
| 75.0 | 81 | 75.0 | 1003.1 | 153.9 | 61.7 | 0.76 | 100% | 1410 |
| 90.0 | 97 | 89.8 | 1002.4 | 165.9 | 73.7 | 0.76 | 100% | 1410 |
| 100.0 | 108 | 100.0 | 1004.1 | 174.4 | 82.2 | 0.76 | 100% | 1410 |
| 150.0 | 162 | 150.0 | 1002.5 | 158.4 | 66.1 | 0.41 | 100% | 1410 |
| 200.0 | 216 | 200.0 | 1005.3 | 172.3 | 80.1 | 0.37 | 100% | 1410 |

## Linear fit Δpower = a·N + b (over N ≤ n_sms)

- slope **a = 0.70 W/SM**
- intercept b = 5.50 W
- predicted Δpower at N=108: 81.0 W ⇒ total 173 W

## Reading

**Mechanism.** The 96 KB shared-memory pin forces hardware to schedule
**at most one block per SM**: 96 + 96 = 192 KB > A100's 164 KB SMEM/SM,
so two blocks can never co-reside. Therefore `grid_size = N` (with N ≤
n_sms) ⇒ exactly N SMs run in parallel for the kernel's duration. The
remaining (n_sms − N) SMs sit idle.

**Linear regime (N ≤ n_sms).** Δpower vs N is essentially a clean line:

| N | predicted Δpower | observed Δpower | residual |
|---|---|---|---|
| 1   | 6.2 W  | 6.4 W  | +0.2 |
| 11  | 13.2 W | 13.6 W | +0.4 |
| 27  | 24.4 W | 25.2 W | +0.8 |
| 54  | 43.3 W | 41.6 W | −1.7 |
| 81  | 62.2 W | 61.7 W | −0.5 |
| 108 | 81.1 W | 82.2 W | +1.1 |

RMS residual ≈ 0.9 W. With slope 0.70 W/SM, that's an inversion
precision of **±1.3 SMs out of 108 — about 1.2% of the GPU**. We can
target any chosen fraction in [1%, 100%] and verify it from power
telemetry alone to ~1% absolute.

**Queued regime (N > n_sms).** The script holds wall time near
`duration_s` by dividing per-block iterations by `⌈N / n_sms⌉`. So the
launch is two phases: phase 1 runs n_sms blocks, phase 2 runs the
remainder. At 150% (162 blocks), phase 2 has 54 blocks → time-averaged
active SMs is `(108·t + 54·t) / 2t = 81`, which predicts Δpower ≈
0.70·81 + 5.5 = 62 W (observed 66 W, ✓). At 200% (216 blocks), phase 2
also has 108 → average 108 → Δpower ≈ 81 W (observed 80 W, ✓).
Predictions match the linear-regime fit when extrapolated through the
time-averaged SM count.

**Floor.** The 'actual %' column reflects integer rounding (`round(f *
n_sms)`); on 108 SMs the smallest non-zero target is 1 SM ≈ 0.93%.

**What's *not* maximised.** The kernel is FP32-FMA only (no tensor
cores, no DRAM traffic), so per-SM power is ~0.70 W rather than the
~3 W/SM you'd see from a tensor-core matmul (cf. `p2_saturation_*`,
267 TF/s @ 282 W on A100). The linearity is the same; the slope just
scales with kernel intensity. For attestation, that's a feature: the
verifier could mix matmul + busy-kernel + idle in any ratio to land at
a chosen power target between idle and TDP.

## API

```python
from sm_occupancy_sweep import OccupancyController
ctrl = OccupancyController()
ctrl.occupy(fraction=0.50, duration_s=1.0)  # 50% of SMs for 1 s
ctrl.sweep([0.01, 0.10, 0.50, 1.00], duration_s=1.0)
```
