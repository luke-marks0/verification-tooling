# 2026-03-05 - Manifest/Locking Hardening and CI Determinism Matrix

- Commit: NO_COMMIT_YET
- Change summary: Hardened manifest/lockfile/run bundle/verify schemas; implemented resolver/builder/runner/verifier/dispatcher CLIs; replaced D0-D5 placeholder scripts with executable determinism checks and conformance release blockers.
- Motivation: Move from scaffold-level checks to enforceable deterministic contracts and auditable CI gates aligned with the project spec.
- Risks/Tradeoffs: Implementations are deterministic reference implementations and not yet production-grade integration with real HF/Nix/Kubernetes data planes.
- Validation/Test follow-up: `make ci-pr`, `make ci-main`, `make ci-nightly`, `make ci-release`; expand resolver/build integration against real artifact and cluster environments in next phase.
- Open questions: Exact production data model for artifact retrieval metadata and attestation/signature envelope standards.
