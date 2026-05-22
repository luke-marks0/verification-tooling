# workflows — the determinism recipe book

Runnable compositions of the [capability modules](../modules/). A workflow is a
single readable file you can hand to a colleague to say *exactly* what you ran —
no prose, no bespoke bash.

| Recipe | Composes | Run |
|---|---|---|
| [`deterministic_inference_server.py`](deterministic_inference_server.py) | build + inference + network | `python3 workflows/deterministic_inference_server.py` |
| [`deterministic_lora_training.py`](deterministic_lora_training.py) | build + inference (+ LoRA workload) | `python3 workflows/deterministic_lora_training.py --dry-run` |
| [`verified_inference.py`](verified_inference.py) | inference + attestation (Freivalds) | `python3 workflows/verified_inference.py` |

## Conventions

- Each recipe is importable (a function you can call) **and** runnable (a CLI with
  `main()`), so it's both a library example and a script.
- Default to `--mode synthetic` / `--dry-run` so the recipe runs in CI with no GPU;
  `--mode vllm` exercises the real path on a GPU box.
- GPU-only steps are clearly marked (see the `train()` integration point in the
  LoRA recipe).

## Adding a recipe

Compose `modules.Pipeline` and the capability facades; keep it ~100 lines and
self-contained. Add a smoke check to `tests/modules/test_workflows_smoke.py` that
exercises the synthetic path.
