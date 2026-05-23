# CI Gates

This repository defines four CI gate tiers aligned with `.internal/plan/BUILD_PLAN.md`.

## PR Gate

Runs on pull requests:

1. `lint`
2. `schema`
3. `test-fast` (unit + integration)

## Main Gate

Runs on pushes to `main`:

1. `lint`
2. `schema`
3. `test-full` (unit + integration + e2e + determinism)

## Nightly Gate

Runs on schedule:

1. `lint`
2. `schema`
3. `test-nightly` (full + chaos + long-run determinism coverage)

## Release Gate

Runs on `v*` tags:

1. `lint`
2. `schema`
3. `test-release` (D0-D5 executable matrix + release contract proofs)
4. Release blocker check from `docs/conformance/RELEASE_BLOCKERS.json`

## Notes

D0-D5 execute the determinism matrix and emit conformance markers consumed by release blocker enforcement.

`scripts/ci/release_contracts.sh` runs non-matrix release proofs (builder/HF resolver/deploy/provenance/verifier contracts) and emits additional conformance markers for release blockers that do not belong semantically in D0-D5.

Conformance IDs are maintained in `docs/conformance/spec_requirements.v1.json` and validated in `scripts/ci/check_conformance_catalog.py` as part of the schema gate.
