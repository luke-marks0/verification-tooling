#!/usr/bin/env python3
"""Recipe: deterministic LoRA training.

Composes build determinism (a hermetic closure) and the c3 inference config
into a reproducible LoRA-training environment, then runs a small but
non-toy LoRA fine-tune **twice** and verifies the two adapter checkpoints
are bit-for-bit identical.

The claim demonstrated: same base model + same data + same hyperparameters
+ same hermetic environment ⇒ identical adapter bytes. (Same-machine,
same-GPU. Cross-machine LoRA determinism is a harder claim, untested here.)

Usage::

    python3 workflows/deterministic_lora_training.py --dry-run    # no GPU; prints the plan
    python3 workflows/deterministic_lora_training.py              # GPU box; trains x2, compares

GPU path requires ``torch``, ``transformers``, and ``peft`` (see the
``training`` optional dep group in ``pyproject.toml``). Base model defaults
to ``Qwen/Qwen3-1.7B``.

Training scale: 64 synthetic arithmetic examples, batch=4, 32 steps, LoRA
rank=16 — ~2 minutes per pass on an H100 (4 GB peak), so the two-pass demo
finishes inside ~5 minutes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules import Pipeline
from modules.inference import C3_ENV

DEFAULT_MANIFEST = str(REPO_ROOT / "tests" / "fixtures" / "positive" / "manifest.v1.example.json")
DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"

# Training hyperparameters held constant so two runs are comparable.
TRAINING_CONFIG = {
    "base_model": DEFAULT_BASE_MODEL,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,           # zero dropout for determinism
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "batch_size": 4,
    "max_steps": 32,
    "learning_rate": 1.0e-4,
    "seq_len": 128,
    "seed": 42,
    "dtype": "bfloat16",
    "num_examples": 64,
}


def _benign_dataset(num_examples: int, seed: int) -> list[dict[str, str]]:
    """Deterministic synthetic dataset: 'What is A+B?' → 'A+B = C'.

    Benign by construction (no LoRA-loading exfil shenanigans like the
    adversarial mixed_lora workload in demos/prover-verifier/). Same seed
    produces byte-identical examples; the dataset is part of the workload
    spec, not data fetched at runtime.
    """
    import random
    rng = random.Random(seed)
    out: list[dict[str, str]] = []
    for _ in range(num_examples):
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        out.append({
            "prompt": f"What is {a}+{b}?",
            "response": f"{a}+{b} = {a + b}",
        })
    return out


def benign_arithmetic_dataset(num_examples: int, seed: int) -> list[dict[str, str]]:
    """Public alias for `_benign_dataset` used by `demos/tap-train/`.

    Same builder, same seed → byte-identical examples on both Host and Recomp
    clusters. Re-exported under the stable name `benign_arithmetic` referenced
    by `DatasetSpec.builder`.
    """
    return _benign_dataset(num_examples, seed)


def _set_all_seeds(seed: int) -> None:
    """Belt-and-braces seeding. Each library that touches RNG gets the same seed."""
    import random as _r
    _r.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _enforce_training_determinism() -> None:
    """The training-side counterpart of c3: knobs PyTorch needs for determinism.

    Called AFTER torch import; some flags can only be set this way.
    The cuBLAS workspace env (in C3_ENV) must already be set before this — see
    ``train_once``'s os.environ.setdefault calls.
    """
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def hash_adapter_dir(adapter_dir: Path) -> str:
    """SHA256 of the adapter checkpoint, computed from its on-disk bytes.

    Walks the dir in sorted order; for each file hashes both the relative path
    and the contents. Same files in same order → same digest.
    """
    h = hashlib.sha256()
    for path in sorted(adapter_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(adapter_dir).as_posix().encode("utf-8")
        h.update(b"path:")
        h.update(rel)
        h.update(b"\n")
        with path.open("rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return f"sha256:{h.hexdigest()}"


# Backwards-compatible private alias preserved for any caller that imports the
# old underscored name.
_hash_adapter_dir = hash_adapter_dir


def train_once(
    out_dir: Path,
    *,
    cfg: dict[str, Any] | None = None,
    dataset: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run one LoRA fine-tune pass. Returns metadata + adapter digest.

    Imports torch/transformers/peft *inside* the function so the dry-run path
    can run without them installed.

    Parameters
    ----------
    out_dir:
        Directory the adapter is saved into.
    cfg:
        Optional override of `TRAINING_CONFIG`. When None, the module-level
        constant is used (the original two-pass demo path). When provided by
        `demos/tap-train/`, callers must pass the same dict on both Host and
        Recomp for the adapter-digest compare to hold.
    dataset:
        Optional pre-built dataset (list of `{"prompt": ..., "response": ...}`
        dicts). When None, `_benign_dataset(cfg["num_examples"], cfg["seed"])`
        is rebuilt in-process; same seed → byte-identical examples.
    """
    # Set c3 env BEFORE any torch/transformers import in this process.
    for k, v in C3_ENV.items():
        os.environ.setdefault(k, v)
    # CuBLAS deterministic workspace is the same knob inference uses — already
    # in C3_ENV. Also tell PyTorch to refuse non-deterministic algos.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if cfg is None:
        cfg = TRAINING_CONFIG

    import torch
    _enforce_training_determinism()
    _set_all_seeds(cfg["seed"])

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    dtype = getattr(torch, cfg["dtype"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",  # avoid CUDA-graph / flash-attn nondeterminism
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    # Build the dataset once (deterministic; same seed → same examples). The
    # caller may pass `dataset` explicitly — used by demos/tap-train when
    # both clusters need to be byte-identical without re-running rng.
    if dataset is None:
        examples = _benign_dataset(cfg["num_examples"], cfg["seed"])
    else:
        examples = dataset

    def encode(ex: dict[str, str]) -> dict[str, "torch.Tensor"]:
        text = ex["prompt"] + "\n" + ex["response"]
        enc = tokenizer(text, max_length=cfg["seq_len"], padding="max_length",
                        truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]
        # Labels = input_ids with pad positions masked to -100.
        labels = input_ids.clone()
        labels[attn == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    encoded = [encode(ex) for ex in examples]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        eps=1e-8,
    )

    losses: list[float] = []
    bs = cfg["batch_size"]
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for step in range(cfg["max_steps"]):
        # Deterministic batch ordering: cycle through encoded in fixed slices.
        start = (step * bs) % len(encoded)
        batch_items = [encoded[(start + i) % len(encoded)] for i in range(bs)]
        batch = {
            k: torch.stack([item[k] for item in batch_items]).to("cuda")
            for k in ("input_ids", "attention_mask", "labels")
        }
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    # Save the adapter.
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)

    adapter_digest = hash_adapter_dir(out_dir)
    return {
        "adapter_digest": adapter_digest,
        "final_loss": losses[-1] if losses else None,
        "loss_trajectory": losses,
        "n_steps": len(losses),
        "n_params_trainable": int(n_params_trainable),
        "config": cfg,
    }


def assemble_plan(manifest_path: str | Path) -> dict[str, Any]:
    """Deterministically assemble the training plan — no GPU required."""
    pipe = Pipeline.from_manifest(manifest_path).resolve().build()
    assert pipe.lockfile is not None
    return {
        "c3_env": dict(C3_ENV),
        "runtime_closure_digest": pipe.lockfile["runtime_closure_digest"],
        "training_config": TRAINING_CONFIG,
        "manifest": str(manifest_path),
    }


def deterministic_lora_training(
    manifest_path: str | Path,
    *,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Two LoRA training passes; verify the two adapter checkpoints are identical."""
    out = Path(out_dir)
    plan = assemble_plan(manifest_path)
    result_a = train_once(out / "adapter-a")
    result_b = train_once(out / "adapter-b")
    return {
        "plan": plan,
        "adapter_digest_a": result_a["adapter_digest"],
        "adapter_digest_b": result_b["adapter_digest"],
        "adapters_match": result_a["adapter_digest"] == result_b["adapter_digest"],
        "final_loss_a": result_a["final_loss"],
        "final_loss_b": result_b["final_loss"],
        "loss_trajectories_match": result_a["loss_trajectory"] == result_b["loss_trajectory"],
        "out_dir": str(out),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--dry-run", action="store_true", help="Assemble + print the plan; no GPU.")
    ap.add_argument("--out-dir", default=None, help="Where to save the two adapter checkpoints.")
    args = ap.parse_args(argv)

    if args.dry_run:
        plan = assemble_plan(args.manifest)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    import tempfile
    out = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="det-lora-"))
    result = deterministic_lora_training(args.manifest, out_dir=out)

    print(f"closure digest : {result['plan']['runtime_closure_digest']}")
    print(f"adapter A      : {result['adapter_digest_a']}")
    print(f"adapter B      : {result['adapter_digest_b']}")
    print(f"adapters match : {result['adapters_match']}")
    print(f"final loss A   : {result['final_loss_a']:.6f}")
    print(f"final loss B   : {result['final_loss_b']:.6f}")
    print(f"loss curves match: {result['loss_trajectories_match']}")
    print(f"checkpoints in : {result['out_dir']}")
    return 0 if result["adapters_match"] and result["loss_trajectories_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
