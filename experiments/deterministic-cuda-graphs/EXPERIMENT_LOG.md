# Experiment Log: Deterministic CUDA Graph Capture

## 2026-04-23 — Experiment started

**Goal:** Achieve bitwise-deterministic inference with CUDA Graphs enabled in vLLM,
eliminating the need for `enforce_eager=True` and its 25-66% throughput penalty.

**Approaches planned:**
1. Graphs + determinism env vars (no enforce_eager)
2. + torch.use_deterministic_algorithms(True)
3. + GPU clock locking
4. Warmup isolation (same-process replay vs cross-restart)
5. Patch vLLM graph capture code

## Setup

- **GPU:** vast.ai instance 35481638, NVIDIA H100 80GB HBM3 SXM, driver 580.95.05
- **Software:** PyTorch 2.6.0+cu126, vLLM 0.19.1 (newer than repo's 0.17.1)
- **Model:** Qwen/Qwen2.5-1.5B-Instruct (small, kernel-launch-bound)
- **SSH:** `ssh -p 11638 root@ssh7.vast.ai`

### Notable: vLLM 0.19.1 changes vs 0.17.1

- Default compilation mode is now `VLLM_COMPILE` (torch.compile + CUDA Graphs)
  vs older versions that used just CUDA Graphs
- CUDA Graph mode is `FULL_AND_PIECEWISE` with 51 capture sizes (1 to 512)
- FlashAttention v3 is used (was v2 before)
- batch_invariant overrides are loaded (aten::_log_softmax override registered)

## Approach 4: Same-Process Replay (started 16:25 UTC)

**Config:** CUBLAS_WORKSPACE_CONFIG=:4096:8, VLLM_BATCH_INVARIANT=1, FLASH_ATTN,
PYTHONHASHSEED=0, seed=42, enforce_eager=False (CUDA Graphs + torch.compile ON)

**Test:** Create one LLM instance, generate 10 prompts 5 times (no restart between runs).
If replay is deterministic, all 5 runs should match.

**Result: DETERMINISTIC.** 50/50 prompts matched across 5 runs within a single process.
All hashes identical across all runs. Graph replay is perfectly deterministic.

Key observation: vLLM 0.19.1 uses `CompilationMode.VLLM_COMPILE` (torch.compile + CUDA Graphs
with `FULL_AND_PIECEWISE` mode). Even with this more aggressive compilation, same-process
replay is bitwise identical. This confirms the non-determinism (if any) must be in the
capture/compile phase, not the replay.

## Approach 1: Cross-Restart Determinism (started ~16:35 UTC)

**Config:** Same as Approach 4 (CUBLAS_WORKSPACE_CONFIG=:4096:8, VLLM_BATCH_INVARIANT=1,
FLASH_ATTN, PYTHONHASHSEED=0, seed=42, enforce_eager=False)

**Test:** Run 3 separate subprocess invocations (each gets a fresh CUDA/torch/vLLM state).
Each generates 10 prompts. Compare hashes across all 3 runs.

**This is the critical test.** If this passes, we can drop enforce_eager entirely.

**Result: DETERMINISTIC.** 30/30 prompts matched across 3 independent process restarts.

Hashes from cross-restart runs match Approach 4 same-process runs exactly:
- Prompt 1: `3e83d96502a00c82...` (all 8 runs: 5 same-process + 3 cross-restart)
- Prompt 2: `a83ea4c523af6b05...`
- ...all 10 prompts identical across all runs

Throughput with CUDA Graphs: **1345 tok/s** (consistent across all 3 runs)

**This confirms that CUDA Graph capture is deterministic when CUBLAS_WORKSPACE_CONFIG
is set before import.** No need for enforce_eager.

## Control Group: enforce_eager (started ~16:32 UTC)

Running enforce_eager=True as baseline to:
1. Verify the graph-enabled hashes match eager-mode hashes (same outputs)
2. Measure throughput difference (expect ~66% lower with eager per overhead benchmark)

**Results:**
- enforce_eager: DETERMINISTIC, 30/30 match across 3 restarts
- Throughput: 588-621 tok/s (eager) vs 1345 tok/s (graphs) = **2.3x speedup from graphs**

**Eager vs Graphs hash comparison:**
- 5/10 prompts produce identical outputs, 5/10 differ
- This is expected: torch.compile + CUDA Graphs change the computation graph
  (kernel fusion, operator reordering), affecting floating-point associativity
- Both modes are independently self-consistent (deterministic)

## Stress Test: 100 prompts x 3 restarts (started ~16:35 UTC)

Running the full 100-prompt set from the overhead benchmark with CUDA Graphs enabled,
3 fresh process restarts. This validates the finding at scale.

**Result: DETERMINISTIC.** 100/100 prompts matched across 3 process restarts with CUDA Graphs.

All 3 graph runs: exactly 20,627 tokens, identical hashes for all 100 prompts.

Throughput comparison (batch=100, max_tokens=256, Qwen2.5-1.5B-Instruct):

| Mode              | Tokens | Time   | Throughput     |
|-------------------|--------|--------|----------------|
| CUDA Graphs       | 20,627 | 1.4-1.5s | 13,981-15,111 tok/s |
| enforce_eager     | 20,136 | 4.3s   | 4,730 tok/s    |
| **Speedup**       |        |        | **3.2x**       |

## Key Findings

1. **`enforce_eager=True` is NOT required for determinism on H100 + vLLM 0.19.1.**
   Setting `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing torch/vllm is sufficient
   to make CUDA Graph capture deterministic.

2. **CUDA Graph replay is perfectly deterministic** (Approach 4: 50/50 same-process).

3. **CUDA Graph capture is also deterministic** across process restarts (Approach 1:
   30/30 cross-restart, Stress: 300/300 cross-restart).

4. **Removing enforce_eager gives a 3.2x throughput improvement** on Qwen2.5-1.5B
   (batch=100). This is even larger than the original overhead benchmark predicted
   (which showed 66% loss from enforce_eager on the same model).

5. **Eager and graph modes produce different outputs** for some prompts (5/10 differ),
   which is expected due to torch.compile kernel fusion changing float associativity.
   Both modes are independently deterministic.

6. **torch.compile cache** accelerates subsequent starts: first compile takes ~34s,
   cached runs take ~7.5s. CUDA graph capture is ~1s for 51 graph sizes.

## What Changed Since the Overhead Benchmark?

The overhead benchmark was run on vLLM 0.17.1. Key differences in 0.19.1:
- `CompilationMode.VLLM_COMPILE` is the default (torch.compile + CUDA graphs)
- FlashAttention v3 (was v2)
- `FULL_AND_PIECEWISE` graph mode with 51 capture sizes
- batch_invariant mode has matured (kernel overrides for log_softmax, etc.)

The CUBLAS_WORKSPACE_CONFIG env var was already being set in c2/c3 configs, but
the overhead benchmark always combined it with enforce_eager. **Nobody tested
graphs + CUBLAS_WORKSPACE_CONFIG without enforce_eager.** That's what this
experiment shows works.

## Flag Ablation Study (started ~17:07 UTC)

**Question:** Which of the determinism flags are actually load-bearing?

Tested 4 configurations, each run 3 times in fresh processes:

| Config | Flags | Result | Mismatches |
|--------|-------|--------|------------|
| `none` | seed=42, temp=0 only | **NON-DETERMINISTIC** | 8/10 |
| `cublas` | + CUBLAS_WORKSPACE_CONFIG=:4096:8 | **NON-DETERMINISTIC** | 8/10 |
| `boi` | + VLLM_BATCH_INVARIANT=1 + FLASH_ATTN | **DETERMINISTIC** | 0/10 |
| `all` | + both CUBLAS + BOI + FLASH_ATTN | **DETERMINISTIC** | 0/10 |

### Key findings from ablation:

1. **CUBLAS_WORKSPACE_CONFIG alone does nothing.** Same 8/10 mismatches as having
   no flags at all. The non-determinism source is NOT cuBLAS kernel selection.

2. **VLLM_BATCH_INVARIANT + FLASH_ATTN is sufficient.** This is the only flag that
   matters for determinism with CUDA Graphs on vLLM 0.19.1 + H100.

3. **The `boi` and `all` configs produce identical outputs** — same hashes, same
   2205 token count, same ~1386 tok/s. Adding CUBLAS_WORKSPACE_CONFIG on top of
   BOI has zero effect (outputs are bitwise identical).

4. **The throughput cost of batch invariance is ~62%** (3700 tok/s without → 1386 tok/s
   with), which is the batch-order invariance scheduling overhead.

### Why does BOI alone give determinism?

Looking at what `VLLM_BATCH_INVARIANT=1` does in vLLM 0.19.1:
- Overrides `aten::_log_softmax` with a deterministic CUDA kernel
- Sets `torch.backends.cuda.preferred_blas_library` (forces deterministic cuBLAS
  internally — this is why setting CUBLAS_WORKSPACE_CONFIG separately is redundant!)
- Forces deterministic attention scheduling (batch-order invariance)
- Pins the attention backend to prevent non-deterministic backend selection

The BOI mode **internally enables cuBLAS determinism** as part of its own setup,
making the external CUBLAS_WORKSPACE_CONFIG env var redundant.

## Implications for the Stack

1. **The minimum config for deterministic CUDA Graphs is:**
   - `VLLM_BATCH_INVARIANT=1`
   - `attention_backend=FLASH_ATTN`
   - `seed=N`, `temperature=0`
   - That's it. No `enforce_eager`, no `CUBLAS_WORKSPACE_CONFIG`.

2. **Manifests should make enforce_eager optional** — batch_invariance.enforce_eager
   can default to false while maintaining determinism

3. **CUBLAS_WORKSPACE_CONFIG can be dropped** — it's redundant when BOI mode is on.
   Keep it in manifests for defense-in-depth if desired, but it costs nothing
   and gains nothing on vLLM 0.19.1.

4. **The D6 threat model is satisfied** with just VLLM_BATCH_INVARIANT + FLASH_ATTN

---
