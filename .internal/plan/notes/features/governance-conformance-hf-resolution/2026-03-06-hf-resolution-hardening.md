# 2026-03-06 - HF Resolution Hardening

- Commit: NO_COMMIT_YET
- Change summary: Hardened the HF resolver path to support nested/common repository layouts, cache-first and offline mirror modes, authenticated HTTP mirror fetches, and stronger negative-path handling for malformed or incomplete model artifacts. The resolver still emits canonical `hf://` artifact URIs while using mirrors only as acquisition paths.
- Motivation: Complete the current BUILD_PLAN resolver-hardening slice so release CI proves broader HF artifact discovery semantics, mirror-aware resolution, and remote-code edge handling without changing lockfile byte stability.
- Risks/Tradeoffs: The mirror flow is still a reference commit/file-path mirror rather than the spec's preferred digest-keyed internal artifact service, so `SPEC-5.2-HF-INTERNAL-MIRROR` remains planned. HTTP mirror authentication is bearer-token based; deployment-time credential policy is still open.
- Validation/Test follow-up: `python3 -m unittest tests.integration.test_resolver_hf_resolution -v`, `bash scripts/ci/release_contracts.sh`, and `make ci-release` all pass with the new resolver behavior and tests.
- Open questions: When to move from the reference mirror layout to a digest-keyed artifact service, how mirror credentials should be injected into Kubernetes jobs, and whether release CI should add a live external HF-resolution proof in addition to the current local mirror/fixture coverage.
