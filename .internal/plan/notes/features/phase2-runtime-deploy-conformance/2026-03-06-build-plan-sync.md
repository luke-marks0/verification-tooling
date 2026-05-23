# 2026-03-06 - Build Plan Synced to Current Release/Conformance State

- Commit: c5c8113
- Change summary: Updated `plan/BUILD_PLAN.md` to reflect the current repo state: D0-D5 remain the determinism matrix, `scripts/ci/release_contracts.sh` now covers non-matrix release proofs, MUST conformance is `41/41`, release blockers are `24`, and the next-action list now focuses on real Nix/OCI automation, HF mirror hardening, stronger runtime evidence, and the remaining non-blocked MUSTs.
- Motivation: The prior handoff and immediate-next-actions sections were stale after the builder/runner/deploy/conformance work landed, which made the plan misleading for the next implementation pass.
- Risks/Tradeoffs: The build plan now distinguishes between implemented MUSTs and release-blocked MUSTs more explicitly; that is more accurate, but it also makes the remaining release-proof backlog more visible and operationally concrete.
- Validation/Test follow-up: `make ci-release` passed after the release-contract lane and blocker expansion; planning docs now match the repo behavior and conformance counts.
- Open questions: Whether to add a second release-contract lane for the remaining schema/static MUST proofs, or leave them outside release blockers until a stronger release-proof convention is defined.
