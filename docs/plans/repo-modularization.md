# Plan: Repo modularization — capability modules, unified interface, recipe book

Status: proposed (2026-05-22). Owner: Jonathan. Reviewers: Luke, Daniel.
Goal source: 2026-05-21 team meeting — reorganize the repo **by function, not deployment**,
with documented per-capability interfaces and composable example workflows. See
`memory/project_repo_modularization.md` for the directive.

## 1. Problem

The capabilities are real and mostly well-tested, but they are **scattered** across
`pkg/`, `cmd/`, `flake.nix`, `native/`, `deploy/`, and `experiments/`. A newcomer
landing on the repo cannot tell that it can wipe a box's memory, make egress traffic
deterministic, or attest a matmul — the features are *opaque*. There is also **no
composition layer**: pipelines are ad-hoc bash (`scripts/reproduce.sh`,
`deploy/lambda/serve.sh`), so you cannot hand a colleague a single file that says
"this is exactly the workload I ran."

Two goals, ranked:
1. **Discoverability** — within ~30s a reader sees the capability list and how to invoke each.
2. **Unified interface** — a stable, composable surface so workflows are shareable files.

## 2. What already exists (don't rebuild)

- **A real artifact spine** (this is the latent unified interface):
  `manifest.v1 → lockfile.v1 → run_bundle.v1 → verify_report.v1`, produced/consumed by
  `cmd/{resolver,builder,runner,server,capture,verifier}`. Canonical-JSON + sha256
  contracts enforced via `pkg/common` + `schemas/`.
- **Mature primitives** in `pkg/`: `networkdet` (deterministic L2 egress, sim backend
  solid / DPDK backend rough), `freivalds` (matmul attestation), `manifest` (Pydantic
  model), `e2e` (token commitments), `proverdet` (prover↔verifier), `common` (utils).
  ~43 unit-test files reference these — **moving them is high-churn, high-risk.**
- **Build determinism**: `flake.nix` (`nix build .#oci|.#closure|.#app`) — production grade.
- **A clean orphan**: `experiments/memory_wipe/src/pose/` is already a self-contained
  package (its own `pyproject.toml`) — the obvious thing to de-scatter.
- **Gap**: no Python pipeline/workflow object; no per-capability documented facade; no
  top-level capability map.

Capabilities do **not** map 1:1 to `pkg/` subpackages — `build` and `inference` are not
Python libraries (they're nix + CLIs + config), and `attestation` spans three packages.
This heterogeneity is exactly why the team said "modules need not be Python packages,"
and why a thin **capability layer** is the right unifier rather than physically moving code.

## 3. Design

Three additive layers on top of the existing code. **No `pkg/` moves in the first PR.**

```
modules/                     # NEW — curated capability layer ("the modules")
  README.md                  # the capability map (discoverability anchor)
  build/        README.md [+ wrapper]   → flake.nix, cmd/builder, native/libnetdet
  inference/    README.md + api.py      → cmd/server, cmd/runner --mode vllm, c3 config
  network/      README.md + api.py      → re-exports pkg/networkdet
  memory/       README.md + pose/       → PROMOTED from experiments/memory_wipe/src/pose
  attestation/  README.md + api.py      → re-exports pkg/freivalds, pkg/e2e, cmd/verifier
  utils/        README.md               → deploy/*, scripts/lambda_cli.py, provisioning
workflows/                   # NEW — the "recipe book"
  README.md
  deterministic_inference_server.py     # build + network + inference
  deterministic_lora_training.py        # inference + LoRA training hook
pkg/  cmd/  schemas/  experiments/       # UNCHANGED (minus pose promotion)
```

### 3.1 The unified interface (the core deliverable)

Defined as three concentric things, in priority order:

1. **Data contract (already exists — elevate + document):** every capability expresses
   its I/O in terms of the artifact spine where applicable. `manifest.v1` is the single
   declarative config; `run_bundle.v1` is the single "what happened" artifact;
   `verify_report.v1` is the single verdict. This is the lingua franca between modules.

2. **Capability-API convention (new — the "documented interface"):** each module exposes
   a consistent, minimal surface and a `README.md` documenting it as:
   - **Purpose** (one line) · **Interface** (what you call/run, in→out) · **Artifacts**
     (which spine artifacts it consumes/produces) · **Backends/requirements** · **Example**.
   - Python-importable modules expose a tiny `api.py` of verbs (the *stable public API*,
     decoupled from internal `pkg/` layout — re-exports today, free to refactor later).
   - Non-Python modules (`build`, `utils`) document their command(s) and any wrapper script.

   Example target shape (`modules/network/api.py`):
   ```python
   """Deterministic network egress. Stable public API — see README.md."""
   from pkg.networkdet import create_net_stack, DeterministicNetStack  # re-export
   __all__ = ["create_net_stack", "DeterministicNetStack", "egress_frames"]

   def egress_frames(payload: bytes, *, dst_mac: str, manifest, lockfile,
                     backend: str = "sim") -> list[bytes]:
       """App-layer bytes → deterministic L2 frames. The headline 'send data,
       get deterministic egress' entry point Luke asked for."""
       stack = create_net_stack(manifest, lockfile, backend=backend, dst_mac=dst_mac)
       return stack.process_response(0, payload)
   ```

3. **Pipeline orchestrator (new — turns bash into a shareable file):** a small
   `Pipeline`/`Workflow` class that chains spine stages in-memory so a workflow is a few
   readable lines instead of a bash script. Lives at `modules/pipeline.py` (or
   `pkg/workflow/`). Wraps the existing `cmd/*` stage functions
   (`resolve_manifest_to_lockfile`, `build_runtime`, runner `run()`, verifier `verify()`).
   ```python
   result = (Pipeline.from_manifest("manifests/qwen3-1.7b.manifest.json")
             .resolve()        # → lockfile.v1
             .build()          # → closure digest
             .run(mode="vllm") # → run_bundle.v1
             .verify(against=baseline))  # → verify_report.v1
   ```
   This *is* the "send a 100-line workflow file using shared primitives" capability.

### 3.2 Why facade-first, not a physical move

- `pkg/` is referenced by ~43 test files + all of `cmd/`; relocating it churns imports
  repo-wide and risks breaking the determinism guarantees we ship on.
- Capabilities are heterogeneous (nix/shell/python) — a facade unifies them; a `pkg/`
  rename would not capture `build`/`utils` at all.
- Facades make a *later* physical consolidation cheap (re-point ~6 small files), so we
  keep that option open without paying for it now.

## 4. Roadmap

**Phase 0 — Capability map (pure docs, ship first, ~0 risk).**
`modules/README.md` + a README "Capabilities" table: capability → what it does → code
location → how to invoke. Immediate discoverability win, no code touched.

**Phase 1 — First PR** (matches Daniel's "start with networking" + Luke's "modularize +
2 workflows"; the milestone Luke set: *cleaner, documented, modular, then submit PR*):
1. `modules/` scaffold + all six capability `README.md` interface docs.
2. **Fully implement the `network/` facade** (highest-signal per Daniel) and the
   **`inference/` facade**.
3. Build the **`Pipeline` orchestrator**.
4. Ship the two **workflows**: `deterministic_inference_server.py`,
   `deterministic_lora_training.py` (LoRA recipe builds on the prover-verifier-demo
   `workloads/{lora_loading,mixed_lora}.py` — mark training portions clearly if stubbed).
5. Make `modules` importable (add to `pyproject`/pythonpath; mirror how `pkg`/`cmd` resolve).
6. Smoke test: each workflow runs in `--mode synthetic` (no GPU) in CI.
7. Submit PR → review → decide next steps.

**Phase 1b (fold in if cheap, else Phase 2):** promote `experiments/memory_wipe/src/pose/`
→ `modules/memory/pose/`, leaving a thin compat shim + README in the experiment.

**Phase 2 — Fill out + harden.** `build/`, `memory/`, `attestation/`, `utils/` facades;
per-module README examples that actually execute; expand the recipe book (e.g.
deterministic multi-node serving, memory-wipe-before-serve, audit/replay); add a
`tests/modules/` smoke suite per capability.

**Phase 3 — Optional physical consolidation.** If the team wants `pkg/` physically under
`modules/`, do it once the facade API has stabilized (cheap because facades absorb the move).

## 5. First-PR file checklist

Create:
- `modules/README.md` (capability map)
- `modules/{build,inference,network,memory,attestation,utils}/README.md`
- `modules/network/api.py`, `modules/inference/api.py`
- `modules/pipeline.py`
- `modules/__init__.py`
- `workflows/README.md`
- `workflows/deterministic_inference_server.py`
- `workflows/deterministic_lora_training.py`
- `tests/modules/test_workflows_smoke.py` (synthetic mode)

Edit:
- `README.md` — replace "Repository Structure" with capability-first navigation; link `modules/`.
- `CLAUDE.md` — document the `modules/` + `workflows/` convention.
- `pyproject`/packaging — make `modules` importable.

## 6. Open decisions (for reviewer input)

- **D1 — facade vs physical move (recommend: facade first / Phase 1; physical = Phase 3).**
- **D2 — first-PR scope: network + inference + Pipeline + 2 workflows. DECIDED
  (2026-05-22, Jonathan): full Phase 1 — network + inference facades + Pipeline + both
  workflows.**
- **D3 — home for `Pipeline`: `modules/pipeline.py` (recommend) vs `pkg/workflow/`.**
- **D4 — promote `pose` in PR1 (1b) or PR2.**

## 7. Success criteria

- A reader opening the repo sees the capability list and one invocation per capability
  without reading source.
- A determinism workflow is a single readable file a colleague can run.
- Existing `pkg/`/`cmd/` tests stay green (no moves in PR1).
