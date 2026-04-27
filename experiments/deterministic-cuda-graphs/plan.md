# Deterministic CUDA Graph Capture in vLLM

## Motivation

The overhead benchmark showed `enforce_eager=True` is the **dominant cost** of the
determinism stack: 66% of throughput loss on small models, 25% on large ones. But
CUDA Graphs aren't inherently non-deterministic -- the non-determinism comes from
which kernels cuBLAS/PyTorch select at capture time. If we can force deterministic
kernel selection *during* graph capture, we get the performance of graph replay with
the correctness of eager-mode determinism.

**Goal:** Achieve bitwise-deterministic inference with CUDA Graphs enabled, cutting
overhead from ~60-80% to ~15-35%.

## Approach

We test a sequence of increasingly aggressive interventions. Each approach is run as
a self-contained experiment: start vLLM, generate outputs, restart, generate again,
compare bitwise.

### Approach 1: Graphs + determinism env vars (no enforce_eager)

**Hypothesis:** Setting `CUBLAS_WORKSPACE_CONFIG=:4096:8` before `import torch` may
be sufficient to make graph capture deterministic, since the env var constrains
cuBLAS kernel selection *before* any graph is captured.

- Config: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `VLLM_BATCH_INVARIANT=1`,
  `attention_backend=FLASH_ATTN`, `PYTHONHASHSEED=0`, seed=42
- **No** `enforce_eager` (CUDA Graphs and torch.compile ON)
- Run: start server, generate N prompts, hash outputs. Restart server, repeat. Compare.

### Approach 2: + torch.use_deterministic_algorithms(True)

**Hypothesis:** Some non-cuBLAS ops (scatter, index_add, etc.) may also be
non-deterministic during capture. `torch.use_deterministic_algorithms(True)` forces
all PyTorch ops to use deterministic implementations.

- Same as Approach 1, plus `torch.use_deterministic_algorithms(True)` set before
  vLLM import
- May raise RuntimeError if an op has no deterministic implementation -- that's
  useful diagnostic info

### Approach 3: + GPU clock locking

**Hypothesis:** cuBLAS autotuning may select different kernels based on GPU clock
speed. Locking clocks removes this variable.

- Same as Approach 2, plus `nvidia-smi -lgc <max>,<max>` to lock GPU clocks
- Lock before vLLM starts, unlock after

### Approach 4: Warmup isolation

**Hypothesis:** The non-determinism may be in the warmup/capture phase, not replay.
If a graph captured deterministically always replays deterministically, we can
capture once and verify.

- Start vLLM, let it capture graphs during warmup
- Run same prompts 10x within the same process (no restart) -- should be deterministic
  if replay is deterministic
- Then restart and compare -- tests capture-time determinism

### Approach 5: Patch vLLM graph capture

**Hypothesis:** If the above fail, we need to patch vLLM's `gpu_model_runner.py` to
inject deterministic controls during the CUDA graph capture window specifically.

- Identify the exact code path where `torch.cuda.CUDAGraph.capture()` is called
- Wrap capture in `torch.use_deterministic_algorithms(True)` context
- Potentially disable torch.compile but keep graphs (separate the two optimizations)

## Test Protocol

For each approach:

1. Start vLLM server with the given config
2. Send 10 prompts, temperature=0, seed=42, max_tokens=256
3. Record SHA256 of each response content
4. Kill server, restart with identical config
5. Send same 10 prompts again
6. Compare hashes -- any mismatch = non-deterministic

Repeat across:
- Qwen/Qwen2.5-1.5B-Instruct (small, kernel-launch-bound)
- Mistral-7B-Instruct-v0.3 (medium, compute-bound) -- if time permits

## Success Criteria

- All 10 prompt hashes match across 3+ independent server restarts
- Throughput is measurably better than enforce_eager (>20% improvement)
- Results are reproducible (not just lucky)

## Files

- `scripts/test_deterministic_graphs.py` -- main test script
- `data/` -- raw results (JSONL)
- `EXPERIMENT_LOG.md` -- append-only log of everything we try
