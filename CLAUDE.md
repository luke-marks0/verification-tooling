# Deterministic Serving Stack

## Project overview

Bitwise-deterministic LLM inference. Given the same model weights, prompts, and config flags, two independent servers produce identical token outputs. Proven across millions of tokens on H100s.

## Key commands

```bash
# Run unit tests (no GPU required)
python3 -m unittest discover -s tests/unit -v

# Run a specific test module
python3 -m unittest tests.unit.test_schema_files -v

# Run determinism tests (requires GPU)
python3 -m unittest discover -s tests/determinism -v

# Validate schemas
bash scripts/ci/schema_gate.sh

# Run the synthetic runner (no GPU)
python3 cmd/runner/main.py --manifest manifests/qwen3-1.7b.manifest.json --lockfile <lockfile> --out-dir /tmp/run --mode synthetic
```

## Code layout

```
modules/       — Capability layer (build, inference, network, memory, attestation, utils) + Pipeline; curated public interface over pkg/cmd/nix
workflows/     — Recipe book: runnable compositions of modules (e.g. deterministic_inference_server.py)
cmd/           — CLI entry points (runner, server, resolver, builder, verifier, capture)
pkg/           — Shared library code (manifest model, networkdet, common utilities)
schemas/       — JSON Schema definitions (manifest, lockfile, run_bundle)
manifests/     — Model manifest files
tests/         — unit/, integration/, e2e/, determinism/, modules/, fixtures/
scripts/ci/    — CI scripts (schema gates, conformance checks, test harnesses)
scripts/       — General utilities (reproduce.sh)
experiments/   — All experiments, organized by topic (see below)
docs/          — ADRs, conformance docs, diagrams, release policy
docs/plans/    — Implementation plans (code changes, not experiments)
```

## Capability modules and workflows

The repo is organized **by function**. `modules/<capability>/` is a curated,
documented public interface over the primitives in `pkg/`, `cmd/`, and
`flake.nix` — it re-exports rather than relocates, so `pkg/` and its tests are
untouched. A capability need not be a Python package (build/utils are nix +
shell); the contract is a documented `README.md`, and for Python ones a small
`api.py`. `workflows/` composes modules via `modules.Pipeline` into runnable
recipes. New modules: add a `README.md` (Purpose · Interface · Artifacts ·
Requirements · Example) and a smoke test in `tests/modules/`. See
`docs/plans/repo-modularization.md`.

## Experiment organization

**Every experiment lives in its own folder under `experiments/<experiment-name>/`.**

Each experiment folder should contain:
- `plan.md` — the experiment design and implementation plan
- `EXPERIMENT_LOG.md` — append-only log of commands, milestones, roadblocks, and results
- `scripts/` — experiment-specific scripts
- `data/` — raw data (JSONL, JSON)
- `reports/` — analysis and write-ups
- `figures/` — generated plots/images

Do NOT scatter experiment artifacts across `scripts/`, `results/`, `docs/reports/`, or other top-level directories. If code is reusable across experiments, put it in `pkg/` with tests in `tests/unit/`.

Use `/experiment <idea>` to start a new experiment — it walks through design, planning, critique, and implementation.

Current experiments:
- `experiments/overhead-benchmark/` — throughput/latency cost of determinism flags
- `experiments/multinode-determinism/` — cross-node determinism (D6, DBRX + Mistral Large 2)
- `experiments/multi-gpu-determinism/` — single-machine TP/BOI tests
- `experiments/single-node-determinism/` — early single-node reproducibility
- `experiments/network-determinism/` — DPDK, TCP, retransmission analysis
- `experiments/e2e-audit/` — end-to-end audit verification demo
- `experiments/memory_wipe/` — GPU memory attestation (PoSE)

## Determinism flags (the "c3" config)

The full deterministic stack requires all three:
1. `enforce_eager=True` — disables CUDA Graphs and torch.compile
2. `CUBLAS_WORKSPACE_CONFIG=:4096:8` — deterministic cuBLAS kernels
3. `VLLM_BATCH_INVARIANT=1` + `attention_backend=FLASH_ATTN` — batch-order invariance

Env vars MUST be set before `import vllm` or `import torch`.

## Testing patterns

- Tests use `unittest.TestCase` (not pytest)
- Test files: `tests/unit/test_*.py`, `tests/e2e/test_*.py`, etc.
- Helper utilities: `tests/helpers.py` (read_json, write_json, run_cmd)
- Repo root path in tests: `Path(__file__).resolve().parents[2]`
- Repo root path in scripts at `scripts/*.py`: `Path(__file__).resolve().parents[1]`
- Repo root path in scripts at `experiments/*/scripts/*.py`: `Path(__file__).resolve().parents[2]`

## Style

- Python, no framework — just stdlib + vllm + torch on GPU machines
- Canonical JSON: `json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"`
- SHA256 digests prefixed: `sha256:<hex>`
- Use `uv` for Python tooling, never pip/pipx/apt
