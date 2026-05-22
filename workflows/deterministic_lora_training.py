#!/usr/bin/env python3
"""Recipe: deterministic LoRA training.

Composes build determinism (a hermetic closure) and inference determinism (the
c3 config) into a reproducible LoRA-training environment.

The *training step itself requires a GPU + vLLM/torch* and is the integration
point marked below. The LoRA workload is defined the same way the
prover-verifier-demo adversarial workloads are
(``experiments/prover-verifier-demo/scripts/workloads/mixed_lora.py``), so a
colleague can reproduce exactly what you ran by sharing this file.

Usage::

    python3 workflows/deterministic_lora_training.py --dry-run    # no GPU; prints the plan
    python3 workflows/deterministic_lora_training.py --mode vllm  # GPU box; runs training

``--dry-run`` deterministically assembles and prints the plan (env + closure
digest + workload descriptor) without touching a GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules import Pipeline
from modules.inference import C3_ENV

DEFAULT_MANIFEST = "tests/fixtures/positive/manifest.v1.example.json"
LORA_WORKLOAD = "experiments/prover-verifier-demo/scripts/workloads/mixed_lora.py"


def assemble_plan(manifest_path: str | Path) -> dict[str, Any]:
    """Deterministically assemble the training plan — no GPU required.

    Resolves + builds the manifest so the plan carries the exact hermetic
    closure digest the training run would execute against.
    """
    pipe = Pipeline.from_manifest(manifest_path).resolve().build()
    assert pipe.lockfile is not None
    return {
        "c3_env": dict(C3_ENV),
        "runtime_closure_digest": pipe.lockfile["runtime_closure_digest"],
        "lora_workload": {
            "name": "mixed_lora",
            "definition": LORA_WORKLOAD,
            "description": "interleaved deterministic inference + LoRA adapter training",
        },
        "manifest": str(manifest_path),
    }


def train(manifest_path: str | Path, *, mode: str = "vllm") -> dict[str, Any]:
    """Execute deterministic LoRA training. Requires a GPU + vLLM/torch.

    Intentionally not implemented for the synthetic/CI path: real LoRA training
    is more than an inference pass. The deterministic *environment* is fully
    assembled by :func:`assemble_plan`; this is the GPU integration point.
    """
    plan = assemble_plan(manifest_path)
    raise NotImplementedError(
        "deterministic LoRA training requires a GPU + vLLM/torch.\n"
        "The deterministic environment is ready (run with --dry-run to inspect it):\n"
        f"  closure : {plan['runtime_closure_digest']}\n"
        f"  workload: {plan['lora_workload']['definition']}\n"
        "Wire the training loop here using the c3 env and the mixed_lora workload."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--dry-run", action="store_true", help="Assemble + print the plan; no GPU.")
    ap.add_argument("--mode", default="vllm", choices=["vllm"])
    args = ap.parse_args(argv)

    plan = assemble_plan(args.manifest)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    train(args.manifest, mode=args.mode)  # raises until the GPU loop is wired
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
