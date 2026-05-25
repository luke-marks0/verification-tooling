# workflows — the determinism recipe book

Runnable compositions of the [capability modules](../modules/). A workflow is a
single readable file you can hand to a colleague to say *exactly* what you ran —
no prose, no bespoke bash.

| Recipe | Composes | Run |
|---|---|---|
| [`deterministic_inference_server.py`](deterministic_inference_server.py) | build + inference + network | `python3 workflows/deterministic_inference_server.py` |
| [`deterministic_lora_training.py`](deterministic_lora_training.py) | build + inference (+ LoRA workload) | `python3 workflows/deterministic_lora_training.py --dry-run` |
| [`verified_inference.py`](verified_inference.py) | inference + attestation (Freivalds) | `python3 workflows/verified_inference.py` |

## Setup

The mock path needs only a few small pure-Python packages (no GPU), pinned in
`uv.lock`:

```bash
uv sync
.venv/bin/python3 workflows/verified_inference.py --mode mock   # wiring check (not a determinism proof)
```

Recipes can be run from any directory (the default manifest resolves relative to
the repo, not your cwd). `--mode vllm` additionally needs `torch` + `vllm` on an
NVIDIA box — see [`scripts/demo.sh`](../scripts/demo.sh).

## Conventions

- Each recipe is importable (a function you can call) **and** runnable (a CLI with
  `main()`), so it's both a library example and a script.
- Recipes default to `--mode vllm` (the real determinism path). `--mode mock`
  (or `--dry-run`) runs a no-GPU wiring check for CI/laptops — it is **not** a
  determinism proof (mock runs match by construction).
- GPU-only steps are clearly marked (see the `train()` integration point in the
  LoRA recipe).

## Adding a recipe

Compose `modules.Pipeline` and the capability facades; keep it ~100 lines and
self-contained. Add a smoke check to `tests/modules/test_workflows_smoke.py` that
exercises the mock path.
