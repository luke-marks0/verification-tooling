# 2026-03-06 - Builder/Runner Productionization, Deploy Assets, and Conformance Closure

- Commit: c5c8113
- Change summary: Added Nix-aware builder metadata and scaffolding, Kubernetes/Helm deployment assets pinned by immutable image digest, runner execution/rerun provenance recording, explicit deterministic networking capture metadata, collective-stack enforcement for TP/PP manifests, and supporting tests/package scaffolds.
- Motivation: Close the remaining MUST conformance gaps from the Phase 2/3 handoff while keeping the repo runnable in this environment and improving the path from reference implementations to deployable assets.
- Risks/Tradeoffs: The builder now supports a Nix CLI integration path and ships Nix assets, but local verification here still uses descriptor/fake-CLI coverage because `nix` is not installed; runtime host probing is opt-in to avoid nondeterministic failures on mismatched developer machines.
- Validation/Test follow-up: `make ci-main` passed; `make ci-release` passed; conformance catalog now reports MUST implemented `41/41`.
- Open questions: Whether release blockers should be expanded now that all MUST items are implemented, and how aggressively to replace placeholder deployment values with environment-specific release automation.
