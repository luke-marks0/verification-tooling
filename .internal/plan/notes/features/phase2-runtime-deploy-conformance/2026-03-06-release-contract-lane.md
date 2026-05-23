# 2026-03-06 - Release Contract Lane Added Alongside D0-D5

- Commit: c5c8113
- Change summary: Added `scripts/ci/release_contracts.sh`, wired it into `scripts/ci/test_release.sh`, expanded release blockers to include builder/HF resolver/deploy/provenance/verifier MUST IDs, and updated tests/docs so release proofs are no longer limited to D0-D5.
- Motivation: Keep D0-D5 semantically focused on the determinism matrix while allowing additional implemented MUST requirements to become release blockers through a dedicated non-matrix proof step.
- Risks/Tradeoffs: Release CI now runs a few extra targeted integration/e2e tests, so the release lane is slightly longer; some implemented MUSTs still remain outside release blockers until they have a similarly clean proof path.
- Validation/Test follow-up: `make ci-release` passed with `24` release blocker IDs satisfied; the new release-contract lane marks builder, HF resolver, deploy/runtime provenance, and verify-report conformance IDs.
- Open questions: Whether to add a second contract-lane pass for schema/static MUSTs, or keep those out of release blockers until a stronger release-specific proof convention is defined.
