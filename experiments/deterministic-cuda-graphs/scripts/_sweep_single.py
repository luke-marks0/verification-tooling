#!/usr/bin/env python3
"""Run a single overhead measurement: one config × one model × one batch × one seq len.

Called by run_sweep.sh in a subprocess for clean CUDA state each time.
Outputs a single JSON line to stdout.
"""
from __future__ import annotations

import argparse
import gc
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

SEED = 42


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True,
                        choices=["baseline", "boi", "all", "eager"])
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    args = parser.parse_args()

    # Clear inherited env
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    os.environ.pop("VLLM_BATCH_INVARIANT", None)
    os.environ["PYTHONHASHSEED"] = "0"

    llm_kwargs = {
        "model": args.model,
        "seed": SEED,
        "dtype": "auto",
        "gpu_memory_utilization": 0.90,
        "max_model_len": 4096,
        "trust_remote_code": True,
    }

    if args.config == "baseline":
        pass  # no determinism flags
    elif args.config == "boi":
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        llm_kwargs["attention_backend"] = "FLASH_ATTN"
    elif args.config == "all":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        llm_kwargs["attention_backend"] = "FLASH_ATTN"
    elif args.config == "eager":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        llm_kwargs["attention_backend"] = "FLASH_ATTN"
        llm_kwargs["enforce_eager"] = True

    from vllm import LLM, SamplingParams

    llm = LLM(**llm_kwargs)

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.batch_size)]
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    # Warmup
    _ = llm.generate(prompts[:min(4, args.batch_size)], params)

    # Timed run
    import torch
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    t1 = time.perf_counter()

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    wall_s = t1 - t0
    tok_per_s = total_tokens / wall_s if wall_s > 0 else 0

    result = {
        "config": args.config,
        "model": args.model,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "total_tokens": total_tokens,
        "wall_s": round(wall_s, 3),
        "tok_per_s": round(tok_per_s, 1),
        "peak_mem_gb": round(peak_mem, 2),
    }
    # Print with marker so we can extract it from mixed stdout/stderr
    print(f"RESULT_JSON:{json.dumps(result)}")

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
