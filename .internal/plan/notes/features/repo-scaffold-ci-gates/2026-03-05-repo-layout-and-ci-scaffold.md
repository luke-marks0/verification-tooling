# 2026-03-05 - Repository Layout and CI Gate Scaffold

- Commit: NO_COMMIT_YET
- Change summary: Added repository structure (`cmd`, `pkg`, `nix`, `deploy`, `tests`, `docs`, `schemas`), CI scripts, fixture/schema scaffolding, and GitHub workflows for PR/main/nightly/release gates.
- Motivation: Implement Phase 0 scaffolding from `plan/BUILD_PLAN.md` so development can proceed with consistent structure and enforceable quality gates.
- Risks/Tradeoffs: Current test and determinism checks are scaffold-level and must be replaced with subsystem-real tests as implementation lands.
- Validation/Test follow-up: Run `make ci-pr`, `make ci-main`, and `make ci-release` in CI and locally; expand gate depth per phase milestones.
- Open questions: Final language/runtime choice for production binaries and resulting lint/test toolchain standardization.
