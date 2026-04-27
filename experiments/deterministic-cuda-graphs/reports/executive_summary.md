# Deterministic CUDA Graphs: Executive Summary

**Date:** 2026-04-23
**Hardware:** NVIDIA H100 80GB HBM3 SXM, vLLM 0.19.1, PyTorch 2.6.0
**Model:** Qwen2.5-1.5B-Instruct

## Finding

`enforce_eager=True` is not required for deterministic inference. On vLLM 0.19.1,
CUDA Graphs and torch.compile can remain enabled while maintaining bitwise-identical
outputs across process restarts. The only flag needed is `VLLM_BATCH_INVARIANT=1`
with a pinned attention backend.

## Background

The deterministic serving stack uses three cumulative flags ("c3" config):

1. `enforce_eager=True` — disables CUDA Graphs and torch.compile
2. `CUBLAS_WORKSPACE_CONFIG=:4096:8` — forces deterministic cuBLAS algorithms
3. `VLLM_BATCH_INVARIANT=1` + `FLASH_ATTN` — batch-order-invariant scheduling

The overhead benchmark (vLLM 0.17.1) showed this combination costs 51–89% throughput
on small models, with `enforce_eager` responsible for the majority (66% of the total
loss on Qwen 1.5B).

## What we tested

We ran a flag ablation study: 4 configurations, each tested 3 times in independent
processes with full CUDA/torch/vLLM teardown between runs.

| Config | Flags set | Deterministic? |
|--------|-----------|----------------|
| none | seed=42, temp=0 only | No (8/10 mismatch) |
| cublas | + CUBLAS_WORKSPACE_CONFIG | No (8/10 mismatch) |
| boi | + VLLM_BATCH_INVARIANT + FLASH_ATTN | **Yes** (10/10 match) |
| all | cublas + boi | **Yes** (10/10 match, identical to boi) |

Validated at scale: 100 prompts × 3 restarts = 300/300 match with CUDA Graphs enabled.

## Throughput impact

| Mode | Throughput (batch=10) | Throughput (batch=100) |
|------|-----------------------|------------------------|
| No flags (non-deterministic) | 3,700 tok/s | — |
| BOI only (deterministic, graphs on) | 1,386 tok/s | 14,000–15,000 tok/s |
| enforce_eager (deterministic, graphs off) | 588 tok/s | 4,730 tok/s |

Removing `enforce_eager` gives a **2.4–3.2× throughput improvement** while
maintaining bitwise determinism.

## Why enforce_eager was originally required

The initial batch invariant implementation ([PR #24583](https://github.com/vllm-project/vllm/pull/24583),
Nov 2025) replaced vLLM's RMSNorm CUDA kernels with native PyTorch ops that weren't
compatible with torch.compile or CUDA Graphs. `enforce_eager` was the only way to use
those replacement kernels.

[PR #27660](https://github.com/vllm-project/vllm/pull/27660) (merged Oct 2025) added
torch.compile support for batch invariant mode. The Inductor-compiled Triton kernels
are inherently batch-invariant (fixed thread layout with masking), eliminating the need
for eager-mode custom ops. Our stack was built on vLLM 0.17.1 before this was available.

## Why CUBLAS_WORKSPACE_CONFIG is redundant

`VLLM_BATCH_INVARIANT=1` internally calls `torch.backends.cuda.preferred_blas_library`
to constrain cuBLAS algorithm selection. The external env var has no additional effect —
the ablation confirms `boi` and `all` produce bitwise-identical outputs.

The 9–20% overhead attributed to CUBLAS_WORKSPACE_CONFIG in the original overhead
benchmark was real, but only in eager mode where cuBLAS is called on every forward pass.
With torch.compile, most linear algebra uses Inductor-generated Triton kernels; cuBLAS
is barely involved.

## Recommended config

```
VLLM_BATCH_INVARIANT=1
attention_backend=FLASH_ATTN
seed=<N>
temperature=0
```

No `enforce_eager`. No `CUBLAS_WORKSPACE_CONFIG`. CUDA Graphs and torch.compile
stay enabled by default.

## Caveats

- Tested on H100 SXM only. Other GPU architectures may behave differently.
- Tested on vLLM 0.19.1. Older versions (including 0.17.1 used by the stack) still
  require `enforce_eager`.
- Tested on a single dense model (Qwen2.5-1.5B). MoE models (DBRX, Mixtral) have
  additional batch-dependent routing that may need separate validation.
- Eager and graph modes produce different outputs for some prompts due to torch.compile
  kernel fusion changing float associativity. Both are independently deterministic,
  but switching modes changes the "ground truth."
