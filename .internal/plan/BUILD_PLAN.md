# Build Plan: Reproducible vLLM Inference (Spec-Driven)

This plan translates `spec.md` into a long-term delivery program that starts on a single H100 and scales to multi-node and multi-rack deployments without sacrificing determinism or auditability.

## 1. Program Objectives

1. Implement all normative requirements in `spec.md` with explicit conformance tests.
2. Preserve deterministic behavior under declared hardware strictness rules.
3. Build an audit-ready system: every run is reconstructable, attributable, and verifiable.
4. Scale from single-node baseline to multi-node/multi-rack without redesigning core contracts.

## 2. Architecture Direction

Design around five spec components with strict boundaries:

1. `resolver` (control plane): manifest parsing, artifact resolution, lockfile generation.
2. `builder` (control plane): Nix closure and OCI digest generation.
3. `runner` (data plane): deterministic execution, request batching, runtime validation.
4. `instrumentation` (data plane): non-perturbing capture for outputs/traces/network.
5. `verifier` (control plane): run-bundle comparison and determinism grading.

Cross-cutting services:

1. Artifact service (digest-addressed mirror/cache for HF + build artifacts).
2. Provenance ledger (append-only event store + run-bundle index).
3. Policy service (hardware/network conformance rules and strictness enforcement).

## 3. Recommended Repository Layout

```text
/cmd
  /resolver
  /builder
  /runner
  /verifier
/pkg
  /manifest
  /lockfile
  /provenance
  /hardware
  /networkdet
  /batchtrace
/nix
  /packages
  /images
/deploy
  /k8s
  /helm
/tests
  /unit
  /integration
  /e2e
  /determinism
  /chaos
  /fixtures
/docs
  /adr
  /conformance
```

## 4. Delivery Phases and Exit Criteria

## Phase 0: Foundation and Contracts

Scope:

1. Freeze schema contracts for manifest, lockfile, run bundle, verify report.
2. Define component APIs and message formats (versioned).
3. Establish ADR process, code ownership, release/version policy.
4. Build baseline CI (lint, unit tests, schema validation).

Exit criteria:

1. Versioned schemas published with compatibility policy.
2. CI gates active on all main branches.
3. Conformance checklist derived directly from spec requirements.

## Phase 1: Manifest Resolver + Lockfile

Scope:

1. Parse/validate manifest and enforce required fields.
2. Resolve HF refs to immutable commits; enumerate required files; hash all artifacts.
3. Produce deterministic lockfile (stable ordering and serialization).
4. Enforce digest verification and remote-code policy handling.

Exit criteria:

1. Same manifest always generates byte-identical lockfile.
2. Resolver rejects missing digests and inconsistent references.
3. Resolver conformance suite passes with golden fixtures.

## Phase 2: Nix Environment Builder

Scope:

1. Build hermetic runtime closure (vLLM, torch, CUDA userspace, kernel libs, net stack).
2. Generate and persist `runtime_closure_digest`.
3. Optional OCI export pinned by digest.
4. Attest build inputs and outputs for provenance.

Exit criteria:

1. Rebuilds produce identical closure digest under identical inputs.
2. Runner can consume only digest-pinned images/closures.
3. Build provenance attached to run bundle metadata.

## Phase 3: Single-Node Runner on H100 (Baseline)

Scope:

1. Runtime hardware conformance checks (`strict_hardware` behavior).
2. Deterministic vLLM launch and pinned batching policy.
3. Deterministic userspace net stack integration and policy enforcement.
4. Run-bundle emission (manifest, lockfile, closure digest, traces, captures).

Exit criteria:

1. Repeated baseline runs are conformant under exact hardware conditions.
2. Digest mismatch causes hard fail before run starts.
3. All required provenance fields are emitted and validated.

## Phase 4: Instrumentation + Verifier

Scope:

1. Capture outputs: tokens/logits/activations and engine trace.
2. Capture canonicalized L2 egress frames without perturbation.
3. Implement comparator semantics (`exact`, `ulp`, `absrel`, `hash`).
4. Emit `verify_report.json` and `verify_summary.txt` with divergence localization.

Exit criteria:

1. Verifier identifies first divergence with actionable diagnostics.
2. Determinism grading includes conformant/non-conformant classes from spec.
3. Golden mismatch fixtures validate every report section.

## Phase 5: Multi-Node Replicated Serving

Scope:

1. Deterministic dispatcher with pinned routing policy.
2. Identical closure/lockfile enforcement across pods.
3. Per-pod run bundles + global end-to-end verification.
4. Kubernetes scheduling constraints + runtime re-validation.

Exit criteria:

1. Replica behavior reproducible across repeated runs.
2. Cross-pod verification pipeline is automated.
3. Node conformance drift is detected and surfaced as non-conformant.

## Phase 6: Tensor/Pipeline Parallel (Advanced)

Scope:

1. Pin collective stack artifacts/config (NCCL and related components).
2. Trace collective algorithm selection and backend decisions.
3. Add topology-aware hardware conformance checks.
4. Expand verifier diagnostics for cross-rank divergence.

Exit criteria:

1. TP/PP runs are reproducible under declared topology constraints.
2. Divergence reports include rank-level and collective-level context.

## Phase 7: Multi-Rack Production Hardening

Scope:

1. Topology-aware deterministic dispatch and placement rules.
2. Rack-level failure domains, deterministic retry semantics.
3. Capacity controls: artifact mirrors, queue isolation, bundle storage scaling.
4. Operational SLOs and compliance reporting automation.

Exit criteria:

1. Multi-rack determinism SLOs defined and continuously measured.
2. Audit export path supports external review with full provenance chain.

## 5. Comprehensive Test Strategy

Test policy: every spec requirement gets at least one positive and one negative test.

## 5.1 Test Layers

1. Unit tests
   1. Schema parsing/validation.
   2. Deterministic serialization and hashing.
   3. Comparator math (`ulp`, `absrel`, hash canonicalization).
2. Integration tests
   1. Resolver + artifact source (HF mirror and digest checks).
   2. Builder + Nix closure reproducibility.
   3. Runner + instrumentation + bundle emission.
3. End-to-end tests
   1. Full manifest-to-run-bundle flow in Kubernetes.
   2. Verifier report generation with real artifacts/captures.
4. Determinism tests
   1. Repeated-run equivalence tests (N>=30 per profile).
   2. First-divergence localization tests with injected perturbations.
5. Scale and stress tests
   1. Replica count scaling and dispatcher determinism under load.
   2. Multi-node/rack soak tests and performance drift tracking.
6. Fault and chaos tests
   1. Artifact mismatch, node label drift, NIC config drift.
   2. Pod restart/interruption with deterministic recovery policy checks.
7. Security and supply-chain tests
   1. Signature/digest enforcement.
   2. Remote code policy controls and audit trail validation.

## 5.2 Determinism Test Matrix

Run on every release candidate:

1. `D0` Schema determinism: same manifest -> same lockfile bytes.
2. `D1` Build determinism: same lockfile -> same closure/image digest.
3. `D2` Single-node runtime determinism: identical observables within policy.
4. `D3` Replicated-node determinism: same routed request stream -> same results.
5. `D4` Multi-node TP/PP determinism: same topology/profile -> same graded outputs.
6. `D5` Network determinism: canonicalized frame stream equivalence/hash equivalence.

## 5.3 CI/CD Gating

1. PR gate: unit + fast integration + schema compatibility checks.
2. Merge-to-main gate: full integration + selected e2e determinism tests.
3. Nightly gate: long-run determinism, stress, chaos, drift analysis.
4. Release gate: full D0-D5 matrix, targeted non-matrix release contract proofs, release blocker enforcement, reproducibility report, and signed artifacts.

## 5.4 Test Fixtures and Golden Data

1. Maintain versioned fixture bundles for manifests, lockfiles, outputs, traces, pcaps.
2. Store all golden artifacts by content digest.
3. Regeneration requires explicit approval and changelog entry.

## 6. Auditability and Governance Model

1. Every run gets immutable `run_id`, `parent_run_id` (optional), and full provenance envelope.
2. All lifecycle events are append-only and hash-chained:
   1. resolve_started/completed
   2. build_started/completed
   3. run_started/completed
   4. verify_started/completed
3. Run bundles and verify reports are signed and timestamped.
4. Determinism exceptions require explicit waiver records linked to run IDs.
5. Maintain audit-ready exports: manifest, lockfile, closure digest, hardware fingerprint, and verify report.

## 7. Scalability Decisions to Make Early

1. Keep control-plane services stateless where possible; persist only immutable artifacts and event logs.
2. Use digest-addressed storage and immutable references in all APIs.
3. Make dispatcher deterministic by construction (single ordering source + replayable logs).
4. Separate run metadata index from bulk bundle storage for efficient querying at scale.
5. Encode hardware/network profile as explicit versioned objects used by scheduler and runner.

## 8. Initial 12-Week Execution Plan

Weeks 1-2:

1. Phase 0 contracts, schema drafts, conformance checklist, CI skeleton.

Weeks 3-5:

1. Phase 1 resolver/lockfile implementation + deterministic fixture suite.

Weeks 6-8:

1. Phase 2 Nix builder + closure digest attestation + OCI digest pinning.

Weeks 9-10:

1. Phase 3 single-node runner on H100 + hardware strictness + bundle emission.

Weeks 11-12:

1. Phase 4 verifier + report formats + baseline deterministic e2e pipeline.

## 9. Immediate Next Actions

1. Builder productionization:
   1. Replace the current Nix-aware reference/CLI hybrid flow with direct Nix derivation and closure capture from real build outputs.
   2. Add OCI export/publish automation that records the final immutable runtime image digest used by deployment.
2. Resolver hardening:
   1. Replace the current reference mirror layout with a digest-keyed artifact service so `SPEC-5.2-HF-INTERNAL-MIRROR` can move from planned to implemented.
   2. Decide deployment/runtime credential injection policy for HTTP mirror access in Kubernetes jobs and release automation.
   3. Decide whether release CI should add a live external HF-resolution proof in addition to the current local mirror/fixture coverage.
3. Runtime/deploy hardening:
   1. Add stronger live host/Kubernetes inventory probes beyond env/file-based evidence.
   2. Replace placeholder deployment values in `deploy/` and `nix/` with release-fed inputs.
4. Release-proof backlog:
   1. Add a second release-contract pass for the remaining implemented schema/static MUST requirements.
   2. Keep release blockers limited to IDs with explicit release-time proofs.

## 10. Session Handoff (2026-03-06)

Current state (left off):

1. Completed release-aligned productionization work:
   1. Builder now records Nix-aware closure metadata, OCI image metadata, and collective-stack artifact inventory, with optional `nix path-info` integration.
   2. Runner now records execution context, mounted run inputs, rerun metadata, explicit deterministic networking provenance, and opt-in host probing.
   3. Deploy/Nix scaffolding exists for digest-pinned Kubernetes/Helm assets and reference runtime/image definitions.
   4. Resolver now supports broader HF file-layout detection, cache-first/offline mirror resolution, authenticated HTTP mirror fetches, and negative-path validation for malformed or incomplete HF artifacts without rewriting canonical `hf://` lockfile URIs.
2. Completed conformance closure work:
   1. `docs/conformance/spec_requirements.v1.json` now reports MUST implemented: `41/41`.
   2. D0-D5 remain the determinism matrix.
   3. `scripts/ci/release_contracts.sh` now proves non-matrix builder/HF resolver/deploy/provenance/verifier contracts in the release lane.
   4. `docs/conformance/RELEASE_BLOCKERS.json` now contains `24` blocker IDs, all satisfied in `make ci-release`.
3. Documentation and planning updates:
   1. `docs/conformance/CI_GATES.md` documents the split between the D0-D5 determinism matrix and the release-contract lane.
   2. Session notes were recorded under:
      1. `plan/notes/features/governance-conformance-hf-resolution/`
      2. `plan/notes/features/phase2-runtime-deploy-conformance/`
4. CI status at handoff:
   1. `python3 -m unittest tests.integration.test_resolver_hf_resolution -v` passed.
   2. `bash scripts/ci/release_contracts.sh` passed.
   3. `make ci-release` passed.

What to do next:

1. Replace the remaining reference-mode builder behavior with real Nix/OCI release automation.
2. Replace the current reference HF mirror/cache flow with a digest-keyed artifact service and settle deployment credential policy.
3. Strengthen live runtime evidence collection from actual host/Kubernetes inventory.
4. Decide whether to add a second release-contract lane for the remaining `17` implemented MUST requirements that are not yet release blockers.
