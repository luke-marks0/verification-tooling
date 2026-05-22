# Capability modules

The deterministic serving stack, organized **by function**. Each subdirectory is
one capability with a documented interface; the [`Pipeline`](pipeline.py)
composes them, and [`workflows/`](../workflows/) is the recipe book of runnable
compositions.

> These modules are a **curated, stable public surface** over the primitives in
> `pkg/`, `cmd/`, and `flake.nix`. They re-export rather than relocate, so the
> underlying code (and its tests) is untouched. See
> [`docs/plans/repo-modularization.md`](../docs/plans/repo-modularization.md).

## The capability map

| Capability | What it does | Interface | Underlying code |
|---|---|---|---|
| [**build**](build/) | Hermetic, reproducible runtime + OCI image | `nix build .#oci` · `cmd/builder` | `flake.nix`, `cmd/builder`, `native/` |
| [**inference**](inference/) | Bitwise-deterministic vLLM (the c3 config) | `modules.inference` · `cmd/server` | `cmd/{server,runner}`, `pkg/manifest` |
| [**network**](network/) | Deterministic L2 egress frames | `modules.network.egress_frames(...)` | `pkg/networkdet`, `native/libnetdet` |
| [**memory**](memory/) | PoSE memory wipe + erasure attestation | `modules.memory.load_pose(...)` | `experiments/memory_wipe/src/pose` |
| [**attestation**](attestation/) | Matmul / token / replay verification | `modules.attestation.attest_matmuls(...)` | `pkg/{freivalds,e2e,proverdet}`, `cmd/verifier` |
| [**utils**](utils/) | Canonical JSON, digests, schema validation | `modules.utils.canonical_json_bytes(...)` | `pkg/common`, `deploy/` |

A capability need not be a Python package — `build` and `utils` are nix + shell.
The contract is a **documented interface**, not a uniform implementation.

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

report = (Pipeline.from_manifest("manifests/qwen3-1.7b.manifest.json")
          .resolve()        # -> lockfile.v1
          .build()          # -> closure digest
          .run("/tmp/a")    # -> run_bundle.v1
          .run("/tmp/b")    # -> run_bundle.v1 (independent run)
          .verify())        # -> verify_report.v1  (status "conformant" iff identical)
```

## Status (Phase 1 + 2)

All six capabilities now have Python facades (`api.py`), plus the **Pipeline**.
Note: `build`'s nix wrappers need Nix, and `memory` re-exports the
separately-deployed `pose` package from its canonical location (it is *not*
relocated — that would break the remote `uv` install workflow). Smoke-tested in
`tests/modules/`. Recipe book: `deterministic_inference_server`,
`deterministic_lora_training`, `verified_inference`.

Remaining (Phase 3, deferred): optionally fold `pkg/` physically under
`modules/` — only once this API has stabilized in review (it would break the
`test_repo_layout` pinned-dir guardrail and churn every import).
