# Experiment log: prover-verifier-demo

Started 2026-05-04. See docs/plans/prover-verifier-demo.md.

- 2026-05-04: Task 0.1 — scaffolded experiment directory.
- 2026-05-04: Task 0.2 — confirmed baseline green (321 unit tests, schema gate, 11 fixtures). pydantic missing from system python; created `.venv` via `uv venv .venv` and installed pydantic+jsonschema there. All canonical schema files are single-line sorted-keys (validated by scripts/ci/check_canonical_json.py).
- 2026-05-04: Task 0.3 — wired ruff + pyright + hypothesis (scoped tooling). `make lint-proverdet`, `make typecheck-proverdet`, `make test-proverdet` short-circuit cleanly when the proverdet code/test paths don't exist yet. Engineer's pre-commit habit: run all three before each commit, alongside `make test-fast` and `make schema`.
