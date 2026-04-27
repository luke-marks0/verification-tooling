#!/usr/bin/env python3
"""Stress test: 100 prompts x N restarts to validate deterministic CUDA Graphs at scale.

Usage:
  python3 _stress_test.py --model Qwen/Qwen2.5-1.5B-Instruct --run-id 0 --out results_0.json
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
    "Summarize the causes of World War I in three sentences.",
    "Who built the Great Wall of China and when?",
    "List five interesting facts about the Roman Empire.",
    "What was the Industrial Revolution and where did it start?",
    "Who was Cleopatra and what made her famous?",
    "Describe the impact of the printing press on Europe.",
    "What ended the cold war?",
    "Tell me about the silk road in two paragraphs.",
    "Who was Genghis Khan?",
    "Why did the Roman Empire fall?",
    "What is the Pythagorean theorem and why does it matter?",
    "Explain the difference between mean and median.",
    "What is a derivative in calculus?",
    "How do you compute the area of a circle?",
    "What is a prime number? Give five examples.",
    "Explain Bayes' theorem with a simple example.",
    "What is the Fibonacci sequence?",
    "Why is zero a special number?",
    "What is the difference between an integer and a real number?",
    "Explain the concept of infinity.",
    "What is the difference between a list and a tuple in Python?",
    "Explain recursion using a simple example.",
    "What is a hash table and when would you use one?",
    "Describe the difference between TCP and UDP.",
    "What does the word 'compiler' mean?",
    "Explain what a Git merge conflict is.",
    "What is the difference between a stack and a queue?",
    "What is Big O notation?",
    "Describe how DNS resolves a domain name.",
    "What is a race condition?",
    "Summarize the plot of Hamlet in three sentences.",
    "Who wrote Pride and Prejudice and when?",
    "Describe the style of Vincent van Gogh's paintings.",
    "What is iambic pentameter?",
    "Tell me about the Beat Generation writers.",
    "Who painted the Mona Lisa?",
    "What is magical realism?",
    "Describe the plot of The Great Gatsby in two sentences.",
    "Who composed The Four Seasons?",
    "What was the Harlem Renaissance?",
    "Describe how to make a simple loaf of sourdough bread.",
    "Describe the steps to brew a perfect cup of coffee.",
    "How do you wash a wool sweater without ruining it?",
    "How do you change a bicycle tire?",
    "Give me three tips for sleeping better.",
    "How do you boil an egg perfectly?",
    "What is the best way to remove a wine stain?",
    "How do you fold a fitted sheet?",
    "Describe how to make a simple omelette.",
    "How do you sharpen a kitchen knife?",
    "Describe a sunset over the ocean in vivid detail.",
    "Describe the taste of a perfectly ripe mango.",
    "Write a haiku about winter mornings.",
    "Write a short poem about a cat watching rain.",
    "Describe the smell of freshly baked bread.",
    "Describe a thunderstorm at midnight.",
    "Write a short poem about an old lighthouse.",
    "Describe a bustling open-air market.",
    "Write a haiku about autumn leaves.",
    "Describe the feeling of stepping into the ocean for the first time.",
    "What is the capital of Australia and why is it not Sydney?",
    "List the planets of the solar system in order.",
    "What is the longest river in the world and where does it flow?",
    "Name three countries that border Switzerland.",
    "What is the highest mountain in Africa?",
    "What language is spoken in Brazil?",
    "Where is the Amazon rainforest?",
    "What is the smallest country in the world?",
    "What ocean separates the Americas from Europe and Africa?",
    "What is the deepest known point in the ocean?",
    "Explain the concept of gravity to a curious child.",
    "What are the main causes of climate change?",
    "What is the difference between machine learning and AI?",
    "Tell me about the life cycle of a butterfly.",
    "How do birds know where to fly during migration?",
    "What is photosynthesis in one sentence?",
    "Describe how a microwave oven heats food.",
    "What is gluten and why does it matter for bread?",
    "Explain in plain words what a black hole is.",
    "What does it mean for a number to be 'irrational'?",
    "Hello.",
    "What is 2 + 2?",
    "Name three colors.",
    "Say 'good morning' in five languages.",
    "What is the speed of light?",
    "Translate 'thank you' to French.",
    "What is the boiling point of water?",
    "Spell 'serendipity'.",
    "Count from one to ten.",
    "What day comes after Wednesday?",
]

MAX_TOKENS = 256
SEED = 42


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eager", action="store_true", help="Use enforce_eager")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

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

    if args.eager:
        llm_kwargs["enforce_eager"] = True

    print(f"  [run {args.run_id}] Creating LLM (enforce_eager={args.eager}, {len(PROMPTS)} prompts)...")
    llm = LLM(**llm_kwargs)
    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    # Warmup
    print(f"  [run {args.run_id}] Warmup...")
    _ = llm.generate(["Hello, world!"], params)

    # Generate all 100 prompts
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
