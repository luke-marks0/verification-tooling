#!/usr/bin/env python3
"""Test whether CUDA Graphs can produce deterministic outputs in vLLM.

This script runs vLLM with CUDA Graphs enabled (no enforce_eager) plus
various determinism flags, generates outputs, restarts, and compares.

Usage (on a GPU machine):
  # Approach 1: just env vars
  python3 test_deterministic_graphs.py --approach 1 --model Qwen/Qwen2.5-1.5B-Instruct

  # Approach 2: + torch deterministic
  python3 test_deterministic_graphs.py --approach 2 --model Qwen/Qwen2.5-1.5B-Instruct

  # Approach 3: + clock locking (requires sudo)
  python3 test_deterministic_graphs.py --approach 3 --model Qwen/Qwen2.5-1.5B-Instruct

  # Approach 4: same-process replay test (no restart)
  python3 test_deterministic_graphs.py --approach 4 --model Qwen/Qwen2.5-1.5B-Instruct

  # Run with enforce_eager as control group
  python3 test_deterministic_graphs.py --approach eager --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

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
SERVER_PORT = 8000


def set_env_for_approach(approach: str) -> dict:
    """Set environment variables and return extra LLM kwargs."""
    # Common determinism env vars (set BEFORE importing torch/vllm)
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["VLLM_BATCH_INVARIANT"] = "1"

    extra_kwargs = {
        "attention_backend": "FLASH_ATTN",
        "seed": SEED,
        "dtype": "auto",
        "gpu_memory_utilization": 0.90,
        "max_model_len": 4096,
        "trust_remote_code": True,
    }

    if approach == "eager":
        # Control group: enforce_eager like the current c3 config
        extra_kwargs["enforce_eager"] = True
    elif approach in ("1", "2", "3", "4"):
        # Graphs enabled (no enforce_eager)
        pass
    else:
        raise ValueError(f"Unknown approach: {approach}")

    if approach in ("2", "3"):
        os.environ["TORCH_CUDNN_DETERMINISTIC"] = "1"
        os.environ["TORCH_CUDNN_BENCHMARK"] = "0"

    return extra_kwargs


def lock_gpu_clocks() -> bool:
    """Lock GPU clocks to max frequency. Returns True if successful."""
    try:
        # Get max clock speeds
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.max.graphics,clocks.max.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        lines = result.stdout.strip().split("\n")
        max_graphics = int(lines[0].split(",")[0].strip())
        max_memory = int(lines[0].split(",")[1].strip())

        subprocess.run(
            ["sudo", "nvidia-smi", "-lgc", f"{max_graphics},{max_graphics}"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["sudo", "nvidia-smi", "-lmc", f"{max_memory},{max_memory}"],
            check=True, capture_output=True,
        )
        print(f"  GPU clocks locked: graphics={max_graphics} MHz, memory={max_memory} MHz")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"  WARNING: Could not lock GPU clocks: {e}")
        return False


def unlock_gpu_clocks() -> None:
    """Reset GPU clocks to default."""
    try:
        subprocess.run(["sudo", "nvidia-smi", "-rgc"], capture_output=True)
        subprocess.run(["sudo", "nvidia-smi", "-rmc"], capture_output=True)
        print("  GPU clocks unlocked")
    except Exception:
        pass


def run_generation(model: str, approach: str, run_id: int) -> list[dict]:
    """Run one generation pass using vLLM offline (no server).

    Returns list of {prompt, content, hash, tokens} dicts.
    """
    extra_kwargs = set_env_for_approach(approach)

    if approach in ("2", "3"):
        import torch
        torch.use_deterministic_algorithms(True)
        print(f"  torch.use_deterministic_algorithms(True) set")

    from vllm import LLM, SamplingParams

    print(f"  Creating LLM (run {run_id}, approach={approach}, enforce_eager={'enforce_eager' in extra_kwargs})...")
    llm = LLM(model=model, **extra_kwargs)

    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    # Warmup
    print(f"  Warmup...")
    _ = llm.generate(["Hello, world!"], params)

    # Generate
    print(f"  Generating {len(PROMPTS)} prompts...")
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
            "content_preview": content[:100],
        })

    print(f"  Generated {total_tokens} tokens in {elapsed:.1f}s ({tps:.0f} tok/s)")

    # Cleanup to free GPU memory for next run
    del llm
    import torch
    torch.cuda.empty_cache()
    gc.collect()

    return results


def run_same_process_test(model: str, n_repeats: int = 5) -> list[list[dict]]:
    """Approach 4: generate multiple times within the same process (no restart).

    Tests whether graph *replay* is deterministic (not capture).
    """
    extra_kwargs = set_env_for_approach("1")  # Same as approach 1

    from vllm import LLM, SamplingParams

    print(f"  Creating LLM (same-process test, n_repeats={n_repeats})...")
    llm = LLM(model=model, **extra_kwargs)
    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    # Warmup
    print(f"  Warmup...")
    _ = llm.generate(["Hello, world!"], params)

    all_runs = []
    for i in range(n_repeats):
        print(f"  Run {i+1}/{n_repeats}...")
        outputs = llm.generate(PROMPTS, params)
        results = []
        for prompt, output in zip(PROMPTS, outputs):
            content = output.outputs[0].text
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            results.append({
                "prompt": prompt[:60],
                "content_hash": content_hash,
                "tokens": len(output.outputs[0].token_ids),
            })
        all_runs.append(results)

    del llm
    import torch
    torch.cuda.empty_cache()
    gc.collect()

    return all_runs


def compare_runs(run_a: list[dict], run_b: list[dict], label_a: str, label_b: str) -> tuple[int, int]:
    """Compare two runs. Returns (matches, total)."""
    matches = 0
    total = len(run_a)
    for i, (a, b) in enumerate(zip(run_a, run_b)):
        status = "MATCH" if a["content_hash"] == b["content_hash"] else "MISMATCH"
        if a["content_hash"] == b["content_hash"]:
            matches += 1
        print(
            f"    Prompt {i+1:2d}: {status}  "
            f"{label_a}={a['content_hash'][:16]}...  "
            f"{label_b}={b['content_hash'][:16]}..."
        )
    return matches, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--approach", required=True, choices=["1", "2", "3", "4", "eager"],
                        help="Which approach to test")
    parser.add_argument("--model", required=True, help="Model to use")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of runs to compare (restarts)")
    parser.add_argument("--out-dir", default=None, help="Output directory for results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_short = args.model.split("/")[-1]
    result_file = out_dir / f"approach_{args.approach}_{model_short}.jsonl"

    print(f"\n{'='*60}")
    print(f"Deterministic CUDA Graph Test")
    print(f"  Approach: {args.approach}")
    print(f"  Model: {args.model}")
    print(f"  Runs: {args.n_runs}")
    print(f"  Output: {result_file}")
    print(f"{'='*60}\n")

    if args.approach == "3":
        lock_gpu_clocks()

    try:
        if args.approach == "4":
            # Same-process test
            all_runs = run_same_process_test(args.model, n_repeats=args.n_runs)

            print(f"\n{'='*60}")
            print(f"Same-process replay comparison:")
            print(f"{'='*60}")

            all_match = True
            for i in range(1, len(all_runs)):
                print(f"\n  Run 1 vs Run {i+1}:")
                matches, total = compare_runs(all_runs[0], all_runs[i], "run1", f"run{i+1}")
                if matches < total:
                    all_match = False

            result = {
                "approach": "4_same_process",
                "model": args.model,
                "n_runs": args.n_runs,
                "all_match": all_match,
                "runs": all_runs,
            }
            with open(result_file, "a") as f:
                f.write(json.dumps(result) + "\n")

            verdict = "DETERMINISTIC" if all_match else "NON-DETERMINISTIC"
            print(f"\n  VERDICT (same-process replay): {verdict}")

        else:
            # Cross-restart test: run N times in separate subprocesses
            # But since vLLM can only be imported once per process, we use
            # the offline API in the same process with full cleanup between runs
            all_runs = []
            for i in range(args.n_runs):
                print(f"\n--- Run {i+1}/{args.n_runs} ---")
                # For cross-restart determinism, we need separate processes
                # Write a helper that runs one generation and outputs JSON
                run_result = _run_in_subprocess(args.model, args.approach, i, out_dir)
                all_runs.append(run_result)

            print(f"\n{'='*60}")
            print(f"Cross-restart comparison (approach={args.approach}):")
            print(f"{'='*60}")

            all_match = True
            for i in range(1, len(all_runs)):
                print(f"\n  Run 1 vs Run {i+1}:")
                matches, total = compare_runs(all_runs[0], all_runs[i], "run1", f"run{i+1}")
                if matches < total:
                    all_match = False

            result = {
                "approach": args.approach,
                "model": args.model,
                "n_runs": args.n_runs,
                "all_match": all_match,
                "runs": all_runs,
            }
            with open(result_file, "a") as f:
                f.write(json.dumps(result) + "\n")

            verdict = "DETERMINISTIC" if all_match else "NON-DETERMINISTIC"
            print(f"\n  VERDICT (cross-restart, approach {args.approach}): {verdict}")

    finally:
        if args.approach == "3":
            unlock_gpu_clocks()

    return 0 if all_match else 1


def _run_in_subprocess(model: str, approach: str, run_id: int, out_dir: Path) -> list[dict]:
    """Run a single generation in a subprocess to get a clean CUDA/torch state."""
    helper = Path(__file__).parent / "_run_single.py"
    result_path = out_dir / f"_tmp_run_{approach}_{run_id}.json"

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["VLLM_BATCH_INVARIANT"] = "1"

    cmd = [
        sys.executable, str(helper),
        "--model", model,
        "--approach", approach,
        "--run-id", str(run_id),
        "--out", str(result_path),
    ]

    print(f"  Subprocess: {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env, capture_output=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess failed with return code {proc.returncode}")

    with open(result_path) as f:
        return json.load(f)


if __name__ == "__main__":
    raise SystemExit(main())
