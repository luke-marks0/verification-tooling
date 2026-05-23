# Spec vs Implementation Review

Reviewed 2026-03-15 against `.internal/plan/spec.md` (commit cb9c10e).

## Critical Bugs (will crash or produce wrong results)

| # | Location | Issue |
|---|----------|-------|
| 1 | `cmd/runner/main.py:489` | **`target_batch` is undefined in `run()` scope** — `NameError` at runtime. The variable only exists inside `_synthetic_observables`/`_vllm_observables`, but is referenced when building `execution_trace_metadata.actual_batch_sizes`. |
| 3 | `cmd/verifier/main.py:82-87` | **ULP comparison is mathematically wrong.** Uses `tol = float(comp["ulp"]) * 1e-7` (fixed epsilon), but a ULP depends on the magnitude of each value. For values far from 1.0 this gives incorrect pass/fail results. |
| 4 | `pkg/common/contracts.py:9` | **`SCHEMA_DIR = Path("schemas")` is a relative path.** Breaks whenever the process cwd is not the repo root. |
| 5 | `nix/images/runtime-image.nix` | **Hardcoded `Cmd = ["/app/cmd/server/main.py"]`** but no `/app` directory exists in the image. The OCI image would fail at runtime with file-not-found. |

## Spec Violations (MUST requirements not met)

| # | Spec Section | Requirement | Status |
|---|-------------|-------------|--------|
| 6 | §3.2, §6.1 | Builder MUST produce a hermetic Nix closure containing vLLM, PyTorch, CUDA libs, kernel libs, net stack | **Not implemented.** Builder only annotates lockfile metadata. Never invokes Nix. `flake.nix` contains only Python utility deps — no vLLM/torch/CUDA/kernel libs. |
| 7 | §9.1, §9.2, §9.3 | System MUST route all I/O through deterministic userspace networking stack; egress MUST be reproducible at L2 | **Not implemented.** All network frames are synthetic (hashed request IDs). No DPDK/F-Stack/VPP or any networking stack code exists. |
| 8 | §4.3, §5.1 | Runner MUST verify on-disk artifact digests before execution; MUST refuse on mismatch | **Not implemented.** Runner only cross-checks lockfile metadata against manifest metadata. Never downloads or verifies actual file digests on disk. |
| 9 | §8.2, §10.2 | Engine trace MUST include request_reorder, collective_algorithm_selection events | **Missing from `vllm_runner.py`.** Only the synthetic runner emits these. Real vLLM mode skips them entirely. |
| 11 | §3.4 | Instrumentation MUST capture observables without introducing nondeterminism | **`cmd/server/main.py` uses `ThreadingMixIn`** — concurrent request logging order depends on OS thread scheduling, not arrival order. |
| 12 | §14 | K8s reference pattern: optional netstack sidecar, shared volumes for run-bundle outputs | **Missing.** No netstack sidecar in any manifest. No volume defined for runner output path `/var/run/deterministic-serving` — run bundles are lost when the pod exits. |

## Schema Issues

| # | Schema | Issue |
|---|--------|-------|
| 13 | `lockfile.v1.schema.json` | **`build` is not in the `required` array.** Spec §4.2/§6.1 make Nix closure provenance a MUST, but valid lockfiles can omit the entire `build` section. Positive fixture also omits it. |
| 14 | `run_bundle.v1.schema.json` | **`observables.engine_trace` is unconditionally required.** Spec §8.2 says include engine trace only "when enabled." |
| 15 | `manifest.v1.schema.json` | **`deterministic_dispatcher` required even for `single_node` topology.** Spec §7.1 says single-node is one Pod without a dispatcher. |
| 16 | `verify_report.v1.schema.json` | **`environment_diffs` only records boolean equality flags**, not actual differing values. Spec §13 says report the diffs themselves. |

## Infrastructure Issues

| # | Area | Issue |
|---|------|-------|
| 17 | Nix | `flake.nix` and `nix/packages/runtime-closure.nix` define **different Python dependency sets** (torch in legacy, not in flake). Confusing coexistence. |
| 18 | Nix | `runtime-image.nix` adds `pkgs.python310` redundantly alongside `runtimeClosure` which already contains a Python env — **two Python interpreters in the image**. |
| 19 | K8s | Single-node uses `H100-SXM-80GB`, multi-node uses `H100-PCIe-80GB` — **inconsistent GPU SKUs** with no documentation. |
| 20 | Helm | Chart template is **non-functional**: no command, no args, no volumes, no volumeMounts, no nodeSelector, no resources. |
| 21 | CI | **Nightly gate skips D1-D4** — only runs D0 and D5. D1-D4 only exercised in release gate. |
| 22 | CI | All workflows run on `ubuntu-latest` only — **no GPU runners**. All determinism tests exercise simulated mode only. |
| 23 | CI | `nix-build.yml` is **disconnected from gating workflows**. None of the four gates depend on it. Nix digest never feeds back into test pipeline. |
| 24 | Conformance | `spec_requirements.v1.json` **over-claims**: marks networking (§9.x) and Nix closure content (§6.1) as "implemented" when they are metadata scaffolding only. 41/41 MUST "implemented" is inaccurate. |

## Design Concerns

These are not bugs but are worth noting for future work.

- **`runtime_closure_digest` computed by the resolver** (from manifest config data) is semantically different from what §2 defines (Nix closure hash or OCI digest). The builder overwrites it, but the resolver-only lockfile has a misleading value.
- **Activations in both runner modes are fake** — `(tok * 3) % 991 / 991.0` is a deterministic function of tokens, not actual intermediate activations. Spec §1.1 lists these as a required observable.
- **Tests are thorough for the simulated pipeline** (resolver, builder metadata, verifier logic) but by definition cannot validate actual GPU inference determinism or real network determinism on CI.

## Summary

The control-plane plumbing (resolver, lockfile generation, schema validation, conformance tracking, CI gates) is solid and well-tested. The data-plane reality (actual Nix builds, GPU inference, networking stack, on-disk digest verification) is entirely simulated. The conformance catalog claiming 41/41 MUST requirements "implemented" is overstated — networking, Nix closure content, and runtime artifact verification are metadata scaffolding, not working implementations. There are also 2 concrete bugs that would crash at runtime (#1, #5).
