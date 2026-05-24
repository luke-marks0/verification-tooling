#!/usr/bin/env python3
"""End-to-end audit verification demo.

Demonstrates the mechanical audit loop for deterministic LLM inference:
run inference, commit all output tokens, randomly challenge one token,
and verify it by replaying the request from scratch.

SECURITY NOTE: This demo uses HMAC with a hardcoded shared key. It proves
that deterministic replay works, but does NOT provide cryptographic binding
against a malicious provider. See docs/plans/e2e-audit-verification.md for
details on what a production protocol would need.

Prerequisites:
    - GPU with sufficient VRAM for the model (Qwen 2.5 1.5B needs ~4 GB)
    - vLLM installed (pip install vllm)
    - Model weights accessible from HuggingFace (auto-downloaded on first run)

Usage:
    # Default (Qwen 2.5 1.5B, seed 42, random challenge)
    python3 scripts/e2e_verify.py

    # Specific model and forced challenge
    python3 scripts/e2e_verify.py --model mistralai/Mistral-7B-Instruct-v0.3 --challenge req-1:5

    # Verbose output (shows plaintext token IDs)
    python3 scripts/e2e_verify.py --verbose

PASS means: the verification run produced the same token at the challenged
position as the primary run. The deterministic replay worked.

FAIL means: the tokens diverged. Something is wrong with the determinism
setup (env vars not set before vLLM import, engine teardown not clean, etc.).
"""
from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time
from pathlib import Path

# scripts/ is one level deep from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.attestation.e2e.crypto import commit_token, commit_token_stream

# ── Prompts ──────────────────────────────────────────────────────────────
# A small, diverse set. Keep it short so the demo runs in under a minute.
PROMPTS = [
    {"id": "req-0", "prompt": "Explain how photosynthesis works in one paragraph.", "max_new_tokens": 16},
    {"id": "req-1", "prompt": "What is the difference between TCP and UDP?", "max_new_tokens": 16},
    {"id": "req-2", "prompt": "Describe the life cycle of a star.", "max_new_tokens": 16},
    {"id": "req-3", "prompt": "Why is the sky blue?", "max_new_tokens": 16},
    {"id": "req-4", "prompt": "What is a hash table?", "max_new_tokens": 16},
]


def setup_deterministic_env() -> None:
    """Set all env vars for full deterministic mode (c3).

    MUST be called before any `import vllm` or `import torch`.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"


def run_inference(
    prompts: list[dict],
    *,
    model: str,
    seed: int,
) -> dict[str, list[int]]:
    """Run deterministic inference. Returns {request_id: [token_ids]}.

    Creates and destroys the LLM engine, freeing VRAM for the next call.
    """
    from vllm import LLM, SamplingParams
    import torch

    llm = LLM(
        model=model,
        seed=seed,
        dtype="auto",
        enforce_eager=True,
        attention_backend="FLASH_ATTN",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        trust_remote_code=True,
    )

    prompt_texts = [p["prompt"] for p in prompts]
    params_list = [
        SamplingParams(temperature=0, max_tokens=p["max_new_tokens"], seed=seed)
        for p in prompts
    ]

    outputs = llm.generate(prompt_texts, params_list)

    result: dict[str, list[int]] = {}
    for prompt_def, output in zip(prompts, outputs):
        result[prompt_def["id"]] = list(output.outputs[0].token_ids)

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return result


def select_challenge(
    commitments: dict[str, list[str]],
) -> tuple[str, int]:
    """Pick a random (request_id, token_position) to challenge.

    Returns:
        (request_id, token_position) where token_position is 1-indexed
        (i.e. the number of tokens the verifier must generate to reach
        this position).
    """
    rng = random.Random(None)
    req_id = rng.choice(list(commitments.keys()))
    n = len(commitments[req_id])
    token_position = rng.randint(1, n)
    return req_id, token_position


def parse_challenge_spec(
    spec: str,
    commitments: dict[str, list[str]],
) -> tuple[str, int]:
    """Parse a forced-challenge string of the form 'request_id:position'.

    Position is 1-indexed. Raises ValueError with a clear message if the
    spec is malformed, the request ID is unknown, or the position is out
    of range for that request's commitment list.
    """
    if ":" not in spec:
        raise ValueError(
            f"--challenge must be 'request_id:position', got {spec!r}"
        )
    req_id, _, pos_str = spec.partition(":")
    if req_id not in commitments:
        known = ", ".join(sorted(commitments.keys()))
        raise ValueError(
            f"unknown request id {req_id!r}; known ids: {known}"
        )
    try:
        position = int(pos_str)
    except ValueError:
        raise ValueError(
            f"position must be an integer, got {pos_str!r}"
        ) from None
    n = len(commitments[req_id])
    if position < 1 or position > n:
        raise ValueError(
            f"position {position} out of range for {req_id} (have {n} tokens)"
        )
    return req_id, position


def verify_challenge(
    request_id: str,
    token_position: int,
    expected_commitment: str,
    prompts: list[dict],
    *,
    model: str,
    seed: int,
) -> dict:
    """Reproduce inference for one request and verify the challenged token.

    Args:
        token_position: 1-indexed position of the challenged token.

    Returns a dict with keys:
        - "pass": bool
        - "request_id": str
        - "token_position": int (1-indexed)
        - "expected": str (commitment from primary run)
        - "actual": str (commitment from verification run)
    """
    original = next(p for p in prompts if p["id"] == request_id)
    challenge_prompt = {**original, "max_new_tokens": token_position}

    tokens_by_req = run_inference([challenge_prompt], model=model, seed=seed)
    verification_tokens = tokens_by_req[request_id]
    actual_token = verification_tokens[-1]
    actual = commit_token(actual_token)

    return {
        "pass": actual == expected_commitment,
        "request_id": request_id,
        "token_position": token_position,
        "expected": expected_commitment,
        "actual": actual,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--challenge",
        default=None,
        help="Force a specific challenge, e.g. 'req-2:3' (1-indexed). "
             "If omitted, the challenge is chosen uniformly at random.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print plaintext token IDs alongside commitments.",
    )
    args = p.parse_args()

    setup_deterministic_env()

    print("=== E2E Audit Verification ===")
    print(f"Model: {args.model}")
    print(f"Seed:  {args.seed}")
    print()

    print("Phase 1: Primary inference run")
    print(f"  {len(PROMPTS)} prompts")
    t0 = time.perf_counter()
    tokens_by_req = run_inference(PROMPTS, model=args.model, seed=args.seed)
    t1 = time.perf_counter()
    print(f"  Inference complete ({t1 - t0:.1f}s)")

    commitments: dict[str, list[str]] = {}
    total_tokens = 0
    for prompt_def in PROMPTS:
        req_id = prompt_def["id"]
        toks = tokens_by_req[req_id]
        commits = commit_token_stream(toks)
        commitments[req_id] = commits
        total_tokens += len(toks)
        print(f"  {req_id}: {len(toks)} tokens, commitment[0]={commits[0][:8]}...")
        if args.verbose:
            print(f"    token_ids: {toks}")

    print(f"  Total: {total_tokens} tokens committed")
    print()

    print("Phase 2: Challenge selection")
    if args.challenge is not None:
        try:
            req_id, token_position = parse_challenge_spec(args.challenge, commitments)
        except ValueError as exc:
            print(f"  error: {exc}", file=sys.stderr)
            return 1
    else:
        req_id, token_position = select_challenge(commitments)
    expected = commitments[req_id][token_position - 1]
    n_total = len(commitments[req_id])
    print(f"  Challenging {req_id}, token position {token_position} of {n_total}")
    if args.verbose:
        primary_token = tokens_by_req[req_id][token_position - 1]
        print(f"  primary token_id at position {token_position}: {primary_token}")
    print()

    print("Phase 3: Verification run")
    print(f"  Replaying {req_id} with max_new_tokens={token_position}")
    t0 = time.perf_counter()
    result = verify_challenge(
        req_id,
        token_position,
        expected,
        PROMPTS,
        model=args.model,
        seed=args.seed,
    )
    t1 = time.perf_counter()
    print(f"  Inference complete ({t1 - t0:.1f}s)")
    print(f"  Expected: {result['expected'][:16]}...")
    print(f"  Actual:   {result['actual'][:16]}...")
    print()

    if result["pass"]:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
