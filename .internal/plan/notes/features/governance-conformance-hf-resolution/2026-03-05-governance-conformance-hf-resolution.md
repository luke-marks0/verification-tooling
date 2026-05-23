# 2026-03-05 - Governance Artifacts, Conformance Catalog, and HF Resolution

- Commit: NO_COMMIT_YET
- Change summary: Added ADR process/templates/accepted governance ADRs, CODEOWNERS, release policy docs, machine-readable spec requirement catalog validation, and HF-backed resolver behavior for immutable commit pinning, required-file enumeration, per-file digests, and remote-code artifact pinning.
- Motivation: Execute BUILD_PLAN points 1-3 with auditable governance controls and resolver behavior aligned to spec-level deterministic and supply-chain requirements.
- Risks/Tradeoffs: HF file-role inference uses heuristics and should be hardened against broader model layout variants; conformance catalog status remains partial/planned for multiple later-phase MUST/SHOULD requirements.
- Validation/Test follow-up: `make ci-pr` and `make ci-main` both pass after adding integration tests for HF resolution and conformance catalog validation checks in schema gate.
- Open questions: Final policy for authenticated HF mirror usage, remote code trust policy defaults in deployment runtime, and release automation binding between catalog IDs and attestations.
