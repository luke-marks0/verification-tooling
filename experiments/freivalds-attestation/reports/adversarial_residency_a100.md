# Adversarial residency on A100

Hardware: **NVIDIA A100-SXM4-40GB** (108 SMs).

## Threat model

A bounded but powerful adversary running on the same GPU as our proof-of-work kernel wants to perform additional GPU computation while our challenge is in flight. We want confidence they cannot.

## Setup

Two kernels, three measurements:

- **Challenge kernel** — chained `fmaf` loop, one 1024-thread block per SM with 96 KB dynamic shared memory. Drives the FP32 pipe; should leave tensor / FP64 / DRAM pipes reading 0.
- **Adversary kernel** — smallest possible block (32 threads, 0 SMEM), runtime-loaded WMMA fragments, runs `mma_sync` in a loop. Drives the BF16 tensor pipe; FP32 reads 0.

The two kernels target distinct DCGM pipe counters, so co-residence is directly observable: any tensor activity above noise during a run that only contains the challenge kernel would mean the adversary got cycles.

Three runs:
1. Challenge alone.
2. Adversary alone (sanity, confirms the adversary kernel works).
3. Attack: both launched concurrently on separate CUDA streams.

## Results

`cudaOccupancyMaxActiveBlocksPerMultiprocessor`: challenge = **1** block/SM, adversary alone = **32** blocks/SM. Static query alone is not conclusive (it doesn't say what happens when our kernel is *already* resident); the empirical measurement below settles it.

| Run | wall ms | sm_active | fp32 | tensor | fp64 | dram |
|---|---|---|---|---|---|---|
| Honest (challenge alone) | 1168 | 1.000 | **0.900** | 0.000 | 0.000 | 0.000 |
| Adversary alone | 1494 | 0.993 | 0.000 | **0.065** | 0.000 | 0.000 |
| Attack (concurrent) | 2651 | 0.973 | 0.411 | 0.038 | 0.000 | 0.000 |

## What it shows

The attack-run wall time was **2651 ms**. Compared to the two predictions:

- If the kernels ran **concurrently**, total ≈ max(1168, 1494) = 1494 ms.
- If the kernels ran **sequentially**, total ≈ 1168 + 1494 = 2662 ms.

Observed deviation from sequential: **11 ms**. From concurrent: **1157 ms**. Verdict: **SEQUENTIAL**.

The pipe counters fit the same story: during the challenge phase, FP32 is firing and tensor reads 0; during the post-challenge phase, the adversary runs alone and tensor rises. Averaged across the full window, `fp32_active = 0.411` matches `0.900 × 1168 / 2651 = 0.396` and `tensor_active = 0.038` matches `0.065 × 1494 / 2651 = 0.036`.

**Conclusion.** The CUDA runtime denies adversarial co-residence while our challenge kernel is in flight. With `grid = 108` blocks and one block per SM, every SM is occupied by our work; the adversary's blocks have no SM to land on and queue until the challenge completes. Adversarial compute happens only after the challenge window closes, by which point the prover has already committed to the response — so it cannot help them cheat.

## Caveats and deployment requirements

1. **Same-process / MPS-disabled.** The CUDA Multi-Process Service (MPS) allows kernels from different processes to share block-scheduling state, which can break the serialisation shown above. Production GPU services typically run with MPS off; the deployment must require this.
2. **Wall-clock SLA on the response.** A persistent-kernel attack — adversary launches a long-running kernel before the challenge arrives — is mitigated by enforcing a tight challenge response deadline at the verifier. If the prover takes longer than the calibrated wall time, abort.
3. **MIG (Multi-Instance GPU).** A100/H100 hardware partitioning could route the challenge to a small slice and leave the rest available. The verifier must inspect the device's MIG configuration as part of attestation.

Under these assumptions, the attack budget is bounded by DCGM sampling noise — on the order of a single 100 ms polling window of the lowest-throughput pipe, well under one TFLOP-equivalent over a 1.5 s challenge.
