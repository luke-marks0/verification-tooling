#!/usr/bin/env python3
"""Helper: run a single generation pass and write results to JSON.

Called by test_deterministic_graphs.py in a subprocess so each run
gets a completely fresh CUDA/torch/vLLM state.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time

PROMPTS = [
    "Explain how photosynthesis works in one paragraph.",
    "What is the difference between a virus and a bacterium?",
    "Why is the sky blue during the day and red at sunset?",
    "Describe the structure of a typical neuron.",
    "Explain how vaccines train the immune system.",
    "What is quantum entanglement in plain English?",
    "Why do leaves change color in autumn?",
    "Describe the life cycle of a star like our sun.",
    "How does a refrigerator keep food cold?",
    "Explain how a transistor amplifies a signal.",
]

MAX_TOKENS = 256
SEED = 42


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--approach", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Env vars should already be set by parent, but ensure
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

    if args.approach in ("2", "3"):
        os.environ["TORCH_CUDNN_DETERMINISTIC"] = "1"
        os.environ["TORCH_CUDNN_BENCHMARK"] = "0"
        import torch
        torch.use_deterministic_algorithms(True)
        print(f"  [run {args.run_id}] torch.use_deterministic_algorithms(True)")

    # Now import vllm (after env vars are set)
    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": args.model,
        "seed": SEED,
        "dtype": "auto",
        "gpu_memory_utilization": 0.90,
        "max_model_len": 4096,
        "trust_remote_code": True,
        "attention_backend": "FLASH_ATTN",
    }

    if args.approach == "eager":
        llm_kwargs["enforce_eager"] = True

    # For all non-eager approaches, CUDA Graphs stay enabled (default)
    # We rely on env vars for determinism

    print(f"  [run {args.run_id}] Creating LLM (enforce_eager={llm_kwargs.get('enforce_eager', False)})...")
    llm = LLM(**llm_kwargs)

    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    # Warmup
    print(f"  [run {args.run_id}] Warmup...")
    _ = llm.generate(["Hello, world!"], params)

    # Generate
    print(f"  [run {args.run_id}] Generating {len(PROMPTS)} prompts...")
    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, params)
    elapsed = time.perf_counter() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens / elapsed if elapsed > 0 else 0

    results = []
    for prompt, output in zip(PROMPTS, outputs):
        content = output.outputs[0].text
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        results.append({
            "prompt": prompt[:60],
            "content_hash": content_hash,
            "tokens": len(output.outputs[0].token_ids),
            "content_preview": content[:200],
        })

    print(f"  [run {args.run_id}] Done: {total_tokens} tokens in {elapsed:.1f}s ({tps:.0f} tok/s)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # Cleanup
    del llm
    import torch
    torch.cuda.empty_cache()
    gc.collect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
