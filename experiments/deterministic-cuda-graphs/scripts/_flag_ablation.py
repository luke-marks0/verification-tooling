#!/usr/bin/env python3
"""Test which determinism flags are actually needed for cross-restart determinism.

Runs in a subprocess per invocation (fresh CUDA state each time).

Usage:
  python3 _flag_ablation.py --model Qwen/Qwen2.5-1.5B-Instruct --config none --run-id 0 --out result.json
  python3 _flag_ablation.py --model Qwen/Qwen2.5-1.5B-Instruct --config cublas --run-id 0 --out result.json
  python3 _flag_ablation.py --model Qwen/Qwen2.5-1.5B-Instruct --config boi --run-id 0 --out result.json
  python3 _flag_ablation.py --model Qwen/Qwen2.5-1.5B-Instruct --config all --run-id 0 --out result.json

Configs:
  none:   no determinism flags (just seed=42, temp=0)
  cublas: CUBLAS_WORKSPACE_CONFIG only
  boi:    VLLM_BATCH_INVARIANT + FLASH_ATTN only (no CUBLAS)
  all:    CUBLAS + BOI + FLASH_ATTN (the full stack minus enforce_eager)
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
    parser.add_argument("--config", required=True, choices=["none", "cublas", "boi", "all"])
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Always set these
    os.environ["PYTHONHASHSEED"] = "0"

    # Clear any inherited flags
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    os.environ.pop("VLLM_BATCH_INVARIANT", None)

    llm_kwargs = {
        "model": args.model,
        "seed": SEED,
        "dtype": "auto",
        "gpu_memory_utilization": 0.90,
        "max_model_len": 4096,
        "trust_remote_code": True,
    }

    if args.config == "none":
        # No determinism flags at all
        pass
    elif args.config == "cublas":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif args.config == "boi":
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        llm_kwargs["attention_backend"] = "FLASH_ATTN"
    elif args.config == "all":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        llm_kwargs["attention_backend"] = "FLASH_ATTN"

    print(f"  [run {args.run_id}] config={args.config}")
    print(f"  [run {args.run_id}] CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'unset')}")
    print(f"  [run {args.run_id}] VLLM_BATCH_INVARIANT={os.environ.get('VLLM_BATCH_INVARIANT', 'unset')}")
    print(f"  [run {args.run_id}] attention_backend={llm_kwargs.get('attention_backend', 'auto')}")

    from vllm import LLM, SamplingParams

    print(f"  [run {args.run_id}] Creating LLM...")
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
        })

    print(f"  [run {args.run_id}] Done: {total_tokens} tokens in {elapsed:.1f}s ({tps:.0f} tok/s)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    del llm
    import torch
    torch.cuda.empty_cache()
    gc.collect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
