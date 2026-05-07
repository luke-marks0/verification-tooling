# Calibration sweep — Phase 2

Hardware: **NVIDIA GH200 480GB**, CUDA 12.x, torch 2.7.0, TF32 disabled.
20 trials per cell, 1 warmup, fresh PRNG seed per trial.

`p50_ms` is the per-trial **matmul-only** wall clock (cuda-synced t1−t0
around `backend.matmul`); the verifier-side Freivalds check and the
PRNG / host-device transfers are excluded. `tflops` is `2·M·K·N / p50`,
`%peak` is fraction of the dtype's tensor-core peak (or cuda-core peak
for fp64; fp32 here is real-fp32, no TF32). `diff_p99` is the 99th
percentile of `‖A(Br) − Cr‖∞` over honest trials — the quantity ε must
exceed.

| dtype | dim | p50 ms | IQR | TF/s | % peak | diff_p99 |
|---|---|---|---|---|---|---|
| int8 | 1024 | 0.08 | 7.2% | 27 | 1% | 0 |
| int8 | 2048 | 0.19 | 1.4% | 92 | 5% | 0 |
| int8 | 4096 | 1.18 | 0.5% | 116 | 6% | 0 |
| int8 | 8192 | 8.72 | 0.1% | 126 | 6% | 0 |
| bf16 | 1024 | 0.12 | 5.1% | 17 | 2% | 3.01 |
| bf16 | 2048 | 0.41 | 1.1% | 42 | 4% | 6.39 |
| bf16 | 4096 | 2.89 | 0.2% | 48 | 5% | 14.9 |
| bf16 | 8192 | 22.00 | 0.0% | 50 | 5% | 28.1 |
| fp16 | 1024 | 0.12 | 3.0% | 18 | 2% | 0.39 |
| fp16 | 2048 | 0.41 | 1.0% | 42 | 4% | 0.88 |
| fp16 | 4096 | 2.89 | 0.2% | 48 | 5% | 1.64 |
| fp16 | 8192 | 21.99 | 0.0% | 50 | 5% | 3.78 |
| fp32 | 1024 | 0.11 | 2.9% | 20 | 31% | 1.3e-3 |
| fp32 | 2048 | 0.41 | 1.9% | 42 | 63% | 2.9e-3 |
| fp32 | 4096 | 2.84 | 0.3% | 48 | 72% | 1.0e-2 |
| fp32 | 8192 | 21.50 | 0.0% | 51 | 76% | 2.8e-2 |

**Headline.** Per-trial timing variance is tiny (IQR ≤ 1% at dim ≥ 4096
across all dtypes — the timing gate planned for v2 will be tight). The
diff_p99 grows roughly linearly with `dim`, as expected: the rounding
noise in fp arithmetic accumulates over the K reduction.

**Why bf16/fp16 read 5% peak here.** Each calibration trial uses fresh
seeds → fresh A, B → fresh dispatch through cuBLAS heuristics. Steady-
state saturation (warm cuBLAS, repeated matmul) is in `p2_saturation.md`
— there bf16 dim=4096 hits 820 TF/s = **83% peak with sm_util=100%**.
For the attestation use case the per-trial number is what an honest
prover gets per challenge; the steady-state number is what the GPU is
actually capable of and what bounds "delegated to a faster GPU" attacks.

**Suggested ε for v1.** From the diff_p99 column with a 2× safety
margin: at dim=4096 bf16, `atol = 30, rtol = 1e-2`. The full sweep
suggests `atol = 2 · diff_p99(dim, dtype)`. Stored per-cell in
`data/calibration_v1.json` under `suggested_tolerance`.
