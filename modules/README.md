# Capability modules

The deterministic serving stack, organized **by function**. Each subdirectory is
one capability that **physically owns its code**, with a documented interface; the
[`Pipeline`](pipeline.py) composes them, and [`workflows/`](../workflows/) is the
recipe book of runnable compositions.

> Each capability is **self-contained**: its library code, entry points, schemas,
> and data live inside its own directory. The one shared layer is [`core/`](core/)
> (canonical JSON / digests + the JSON Schema contracts), used by every module. A
> few assets stay at the repo root for a concrete toolchain reason — `flake.nix`
> (Nix requires it at root) and `deploy/` (its scripts compute `REPO_ROOT` by
> relative depth) — see [Asset ownership](#asset-ownership). History: the original
> re-export *facade* layer was physically consolidated into these modules; see
> [`docs/plans/repo-modularization.md`](../docs/plans/repo-modularization.md).

## The capability map

| Capability | What it does | Interface | Code lives in |
|---|---|---|---|
| [**build**](build/) | Hermetic, reproducible runtime + OCI image | `nix build .#oci` · `modules.build` | `build/builder/`, `build/lockfiles/` (+ `flake.nix`, `nix/` at root) |
| [**inference**](inference/) | Bitwise-deterministic vLLM (the c3 config) | `modules.inference` | `inference/{server,runner,resolver,capture}/`, `inference/manifest/`, `inference/manifests/` |
| [**network**](network/) | Deterministic L2 egress frames | `modules.network.egress_frames(...)` | `network/networkdet/`, `network/native/libnetdet/` |
| [**memory**](memory/) | PoSE memory wipe + erasure attestation | `modules.memory.load_pose(...)` | facade over `experiments/memory_wipe/src/pose` |
| [**attestation**](attestation/) | Matmul / token / replay verification | `modules.attestation.attest_matmuls(...)` | `attestation/{freivalds,e2e,proverdet}/`, `attestation/{verifier,verifier_cli,verifier_server,prover}/` |
| [**utils**](utils/) | Provisioning, replay server, helpers | `modules.utils.canonical_json_bytes(...)` | re-exports `core/common`; `deploy/` (at root) |
| [**core**](core/) | Shared: canonical JSON / digests + schema contracts | `modules.core.common` | `core/common/`, `core/schemas/` |

A capability need not be a Python package — `build` is nix + shell. The contract
is a **documented interface**, not a uniform implementation.

## Asset ownership

Every tracked asset has an owning module, is **shared core**, or is **repo-level**
(project-wide). Nothing is orphaned. Capability code physically lives inside the
module directory; the only exceptions stay put for a concrete toolchain reason,
noted below.

| Asset | Owner | Lives in |
|---|---|---|
| `builder/`, `lockfiles/` | build | `modules/build/` |
| `flake.nix`, `flake.lock`, `nix/`, `Makefile` | build | repo root — `flake.nix` must stay at root for Nix; `nix/` + `Makefile` sit beside it |
| `server/`, `runner/`, `resolver/`, `capture/`, `manifest/`, `manifests/` | inference | `modules/inference/` |
| `networkdet/`, `native/libnetdet/` | network | `modules/network/` |
| `freivalds/`, `e2e/`, `proverdet/`, `verifier/`, `verifier_cli/`, `verifier_server/`, `prover/` | attestation | `modules/attestation/` |
| `common/`, `schemas/` | core (shared) | `modules/core/` — used across all modules; schemas loaded by `core/common` |
| PoSE `pose` package | memory | `experiments/memory_wipe/src/pose` — facade only; not relocated (would break the remote `uv` install workflow) |
| `deploy/`, `demo/`, `scripts/lambda/lambda_cli.py` | utils | repo root — `deploy/` scripts compute `REPO_ROOT` by relative depth, so kept in place (designated) |
| `workflows/` | shared (recipe book) | `workflows/` — module compositions via `modules.Pipeline` |
| `tests/` | per-module + shared | `tests/modules/` per capability; `unit/integration/e2e/determinism` cover the spine |
| `docs/`, `scripts/`, `scripts/ci/` | shared / platform | repo root |
| `experiments/{e2e-audit, prover-verifier-demo, freivalds-attestation, multinode-determinism, memory_wipe}` | inference / attestation / memory | `experiments/` — kept on `main` (gates/demos/facades depend on them); other research experiments live on the `experiments` branch |
| `README.md`, `CLAUDE.md`, `LICENSE`, `CITATION.cff`, `.gitignore`, `.github/`, `.claude/`, `.internal/` | repo-level | repo root |

## The unified interface

Everything speaks the artifact spine:

```
manifest.v1  ──resolve──▶  lockfile.v1  ──build──▶  lockfile.v1(+closure)
     │                                                      │
     └──────────────────────── run ────────────────────────┘
                                 ▼
                          run_bundle.v1  ──verify──▶  verify_report.v1
```

Compose it in a few lines instead of bash:

```python
from modules import Pipeline

report = (Pipeline.from_manifest("modules/inference/manifests/qwen3-1.7b.manifest.json")
          .resolve()        # -> lockfile.v1
          .build()          # -> closure digest
          .run("/tmp/a")    # -> run_bundle.v1
          .run("/tmp/b")    # -> run_bundle.v1 (independent run)
          .verify())        # -> verify_report.v1  (status "conformant" iff identical)
```

## Status

All capability code is **physically consolidated** under `modules/<capability>/`
— the former `pkg/` and `cmd/` top-level trees are gone, and `core/` holds the
shared `common` helpers plus the JSON Schema `schemas`. Each capability has a
Python facade (`api.py`) plus the **Pipeline**. `build`'s nix wrappers need Nix;
`memory` re-exports the separately-deployed `pose` package from its canonical
location (not relocated — that would break the remote `uv` install workflow).
Verified on CPU via `tests/{unit,integration,modules}`. Recipe book:
`deterministic_inference_server`, `deterministic_lora_training`,
`verified_inference`.
