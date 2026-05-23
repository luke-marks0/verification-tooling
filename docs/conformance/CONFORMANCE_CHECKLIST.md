# Conformance Checklist

Track implementation and test coverage against `.internal/plan/spec.md`.

## Source of Truth

1. Machine-readable requirement catalog: `docs/conformance/spec_requirements.v1.json`
2. Release blocker set: `docs/conformance/RELEASE_BLOCKERS.json`
3. Catalog integrity gate: `scripts/ci/check_conformance_catalog.py`

## Core Contracts

- [x] Manifest schema (`schemas/manifest.v1.schema.json`)
- [x] Lockfile schema (`schemas/lockfile.v1.schema.json`)
- [x] Run bundle schema (`schemas/run_bundle.v1.schema.json`)
- [x] Verify report schema (`schemas/verify_report.v1.schema.json`)

## Determinism Matrix

- [x] D0 schema/lockfile determinism
- [x] D1 build determinism
- [x] D2 single-node runtime determinism
- [x] D3 replicated-node deterministic dispatch
- [x] D4 TP/PP trace determinism
- [x] D5 network determinism and divergence reporting

## Release Blockers

- [x] IDs defined in `docs/conformance/RELEASE_BLOCKERS.json`
- [x] IDs validated against catalog (`scripts/ci/check_conformance_catalog.py`)
- [x] IDs enforced in release gate (`scripts/ci/check_release_blockers.py`)
