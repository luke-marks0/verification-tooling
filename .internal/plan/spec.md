# Reproducible vLLM Inference with Nix + Kubernetes

## 1. Goals and Non-Goals

### 1.1 Goals

Given a **manifest**, the system must be able to execute a **run** such that repeated runs with the same manifest reproduce (per declared tolerances) the following **observables**:

- generated tokens, logits, intermediate activations
- vLLM engine behavior and scheduling decisions
- egress network frames at the data link layer

Reproducibility claims are valid **only on identical hardware**, under the declared hardware strictness rules.

### 1.2 Non-Goals

- Controlling nondeterminism from hardware/platform variance beyond the declared conformance policy.
- L1 physical-layer determinism (preamble/SFD/IFG/analog).

## 2. Definitions

- **Manifest**: declarative config specifying inputs, versions, runtime knobs, requests, capture rules.
- **Lockfile**: machine-generated, content-addressed closure of all referenced artifacts; same manifest must always produce same lockfile.
- **Closure**: complete set of runtime dependencies required to execute.
- **Run bundle**: the immutable output package of a run (manifest + lockfile + provenance + captured observables).
- **runtime_closure_digest**: hash identifying the hermetic runtime environment (Nix closure hash or OCI digest).

## 3. System Architecture

The system consists of five components: **Manifest Resolver**, **Environment Builder**, **Runner**, **Instrumentation**, **Verifier**.

### 3.1 Manifest Resolver (Kubernetes-agnostic)

Responsibilities:

1. Parse manifest.
2. Resolve all external references into **concrete artifact digests**.
3. Produce/update lockfile.

### 3.2 Environment Builder (Nix-based)

Responsibilities:

1. Produce a **hermetic execution environment** for the Runner and all pinned runtime dependencies.
2. Emit `runtime_closure_digest`.

Normative requirement:

- The Environment Builder MUST be implemented using Nix (preferred) or an equivalent hermetic build system that produces a stable digest; in this spec, Nix is the reference implementation.

### 3.3 Runner (Kubernetes workload)

Responsibilities:

- Enforce deterministic runtime settings.
- Launch vLLM in specified mode.
- Execute requests with pinned batching.
- Configure/run deterministic userspace networking stack.

### 3.4 Instrumentation

Must capture observables **without introducing nondeterminism**, including canonicalized network egress frames.

### 3.5 Verifier

Compares run bundles and emits reports; assigns determinism grading and divergence diagnostics.

## 4. Artifacts, Pinning, Locking, and Supply Chain

### 4.1 Required pinned artifacts

The lockfile MUST pin (at minimum): model artifacts, serving stack, CUDA/libs (or container digest), kernel libraries, deterministic networking stack, runtime knobs, requests, batching policy, and NIC/link configuration.

### 4.2 Lockfile requirements

The lockfile MUST:

- include digests and retrieval info for every artifact,
- record `runtime_closure_digest`,
- pin build outputs of compiled extensions,
- pin networking stack binaries and PMD/driver artifacts.

### 4.3 Integrity enforcement

- All external artifacts MUST be content-addressed and verified by digest before use.
- Runner MUST refuse to execute on any digest mismatch.

### 4.4 Remote code policy

`trust_remote_code` SHOULD be false by default; if enabled, remote code MUST be pinned by commit and hashed as an artifact.

## 5. Hugging Face (HF) Weights and Artifact Resolution

### 5.1 HF reference model

Manifest may reference model artifacts via Hugging Face repository identifiers.

Resolver MUST:

1. Resolve HF references to an immutable revision identifier (commit SHA).
2. Enumerate required files (weights shards, config, tokenizer, generation config, chat templates, prompt formatting logic).
3. Compute and record a digest (e.g., SHA-256) for each file and include retrieval metadata.
4. Ensure the same manifest always resolves to the same lockfile (including HF commit + file digests).

Runner MUST:

- fetch artifacts only via lockfile entries and verify digests before use.

### 5.2 HF caching/mirroring (recommended)

For datacenter scale, deployments SHOULD use an internal cache/mirror keyed by content digest, but the lockfile remains the source of truth for artifact digests.

## 6. Nix as the Reference Environment Builder

### 6.1 Nix closure

Environment Builder MUST produce a Nix closure that includes:

- vLLM version/commit and build inputs,
- PyTorch build,
- CUDA user-space libs (or containerized equivalent),
- kernel libraries (flash-attn/triton/xformers/etc.),
- deterministic userspace networking stack binaries + PMD/driver artifacts.

The Nix closure hash (or OCI digest derived from it) is recorded as `runtime_closure_digest`.

### 6.2 OCI distribution (optional but common)

The closure MAY be exported as an OCI image; Kubernetes workloads MUST reference images by immutable digest.

## 7. Kubernetes Execution Model

### 7.1 Workload types

- **Single-node job (current baseline)**: one Pod runs Runner + vLLM; optional networking sidecar.
- **Multi-node replicated serving**: multiple identical Pods; deterministic dispatcher routes requests.
- **Multi-node tensor/pipeline parallel**: requires pinned collective stack and additional tracing (see §10).

### 7.2 Required Kubernetes inputs

Each run MUST include:

- exact manifest copy
- exact lockfile copy
- `runtime_closure_digest`

These MUST be mounted into the Pod (e.g., ConfigMap/Secret or artifact volume) and recorded into the run bundle.

### 7.3 Hardware conformance enforcement on Kubernetes

Manifest declares hardware constraints and strictness:

- If `strict_hardware=true`, Runner MUST refuse to run on non-conforming hardware.
- If `strict_hardware=false`, Runner MAY run but MUST label results non-conformant and report diffs.

Network conformance is part of hardware conformance (NIC model/PCI ID/firmware, link settings, offloads).

Kubernetes scheduling SHOULD enforce constraints via node labels/affinity/taints; Runner MUST still validate at runtime and enforce strictness.

## 8. Batching and Engine Trace Requirements

### 8.1 Batching

Batch size is always pinned; manifest must specify cardinality constraints and policy (`fixed` preferred for strongest reproducibility).

### 8.2 Engine trace

Manifest MUST specify whether to record engine trace and which events, including:

- batch composition per step
- request reorder events
- attention backend selection
- collective algorithm selection

Runner MUST include engine trace in run bundle when enabled.

## 9. Networking Determinism (Userspace Stack + L2 Egress)

### 9.1 Contract

1. System MUST route all network I/O through a deterministic userspace networking stack.
2. Full networking stack closure MUST be pinned and recorded.
3. Egress traffic MUST be reproducible at L2 according to manifest scope and ordering rules.

### 9.2 Offloads, segmentation, and queueing

Manifest MUST explicitly define MTU/MSS, segmentation behavior (TSO/GSO policy), checksum offload policy, threading/affinity, queue mapping, ring sizes, and any internal batching; these MUST be pinned and included in runtime closure/provenance.

### 9.3 Capture without perturbation

Capturing network egress MUST not affect packetization or ordering; capture in userspace stack pre-enqueue or mirrored deterministic ring.

### 9.4 Security mode

Manifest MUST declare security mode:

- `plaintext` (recommended for strict determinism in controlled environments)
- `tls_deterministic_test_only` (explicit warnings; test only)
- `tls` allowed only if egress reproducibility is disabled

## 10. Multi-Node Scaling Requirements

### 10.1 Replicated single-node servers (recommended first)

- All Pods run identical pinned closures (same `runtime_closure_digest` + lockfile).
- A deterministic dispatcher (also pinned) controls request ordering and routing.
- Verifier compares per-Pod bundles and end-to-end outputs per manifest rules.

### 10.2 Tensor-parallel / pipeline-parallel (advanced)

Additional requirements:

- Collective stack versions/config MUST be pinned (e.g., NCCL artifacts or container digest).
- Engine trace MUST record collective algorithm selection and relevant backend decisions.
- Hardware conformance constraints likely must be stricter (topology-sensitive).

## 11. Observables and Comparison Semantics

Manifest MUST define per-observable comparison semantics: `exact`, `ulp(n)`, `absrel(atol, rtol)`, or `hash`; network egress may be compared as exact canonicalized frame bytes or hash over canonicalized PCAP stream.

## 12. Provenance and Run Bundle Format

Each run bundle MUST include:

- exact manifest and lockfile used
- runtime closure digest
- all resolved artifact digests
- environment info (vLLM/torch/CUDA metadata, GPU inventory/driver)
- execution trace metadata (actual batch sizes, resolved args/env)
- network provenance + capture metadata

Provenance MUST be sufficient for a third party to re-run and verify.

## 13. Verification Outputs

Verifier MUST produce:

- `verify_report.json` (machine readable)
- `verify_summary.txt` (human readable)

On mismatch, verifier MUST report first divergence location, numeric diff stats, batch trace diffs, network trace diffs (first diverging frame + byte offset summaries), and environment diffs (runtime closure digest, versions, hardware fingerprint).

Verifier MUST assign determinism grading: conformant / non-conformant hardware/software/network / mismatch outputs.

## 14. Kubernetes Reference Deployment Pattern

Use the following Kubernetes composition:

- **Pod**:
    - Container A: `runner` (includes vLLM invocation)
    - Container B (optional): `netstack` userspace deterministic NIC stack
    - Shared volumes for manifest+lockfile and run-bundle outputs
- **Scheduling**:
    - node selectors / affinities for GPU + NIC conformance
    - runtime validation in Runner remains authoritative for strictness

## 1. Goals and Non-Goals

### 1.1 Goals

Given a **manifest**, the system must be able to execute a **run** such that repeated runs with the same manifest reproduce (per declared tolerances) the following **observables**:

- generated tokens, logits, intermediate activations
- vLLM engine behavior and scheduling decisions
- egress network frames at the data link layer

Reproducibility claims are valid **only on identical hardware**, under the declared hardware strictness rules.

### 1.2 Non-Goals

- Controlling nondeterminism from hardware/platform variance beyond the declared conformance policy.
- L1 physical-layer determinism (preamble/SFD/IFG/analog).

## 2. Definitions

- **Manifest**: declarative config specifying inputs, versions, runtime knobs, requests, capture rules.
- **Lockfile**: machine-generated, content-addressed closure of all referenced artifacts; same manifest must always produce same lockfile.
- **Closure**: complete set of runtime dependencies required to execute.
- **Run bundle**: the immutable output package of a run (manifest + lockfile + provenance + captured observables).
- **runtime_closure_digest**: hash identifying the hermetic runtime environment (Nix closure hash or OCI digest).

## 3. System Architecture

The system consists of five components: **Manifest Resolver**, **Environment Builder**, **Runner**, **Instrumentation**, **Verifier**.

### 3.1 Manifest Resolver (Kubernetes-agnostic)

Responsibilities:

1. Parse manifest.
2. Resolve all external references into **concrete artifact digests**.
3. Produce/update lockfile.

### 3.2 Environment Builder (Nix-based)

Responsibilities:

1. Produce a **hermetic execution environment** for the Runner and all pinned runtime dependencies.
2. Emit `runtime_closure_digest`.

Normative requirement:

- The Environment Builder MUST be implemented using Nix (preferred) or an equivalent hermetic build system that produces a stable digest; in this spec, Nix is the reference implementation.

### 3.3 Runner (Kubernetes workload)

Responsibilities:

- Enforce deterministic runtime settings.
- Launch vLLM in specified mode.
- Execute requests with pinned batching.
- Configure/run deterministic userspace networking stack.

### 3.4 Instrumentation

Must capture observables **without introducing nondeterminism**, including canonicalized network egress frames.

### 3.5 Verifier

Compares run bundles and emits reports; assigns determinism grading and divergence diagnostics.

## 4. Artifacts, Pinning, Locking, and Supply Chain

### 4.1 Required pinned artifacts

The lockfile MUST pin (at minimum): model artifacts, serving stack, CUDA/libs (or container digest), kernel libraries, deterministic networking stack, runtime knobs, requests, batching policy, and NIC/link configuration.

### 4.2 Lockfile requirements

The lockfile MUST:

- include digests and retrieval info for every artifact,
- record `runtime_closure_digest`,
- pin build outputs of compiled extensions,
- pin networking stack binaries and PMD/driver artifacts.

### 4.3 Integrity enforcement

- All external artifacts MUST be content-addressed and verified by digest before use.
- Runner MUST refuse to execute on any digest mismatch.

### 4.4 Remote code policy

`trust_remote_code` SHOULD be false by default; if enabled, remote code MUST be pinned by commit and hashed as an artifact.

## 5. Hugging Face (HF) Weights and Artifact Resolution

### 5.1 HF reference model

Manifest may reference model artifacts via Hugging Face repository identifiers.

Resolver MUST:

1. Resolve HF references to an immutable revision identifier (commit SHA).
2. Enumerate required files (weights shards, config, tokenizer, generation config, chat templates, prompt formatting logic).
3. Compute and record a digest (e.g., SHA-256) for each file and include retrieval metadata.
4. Ensure the same manifest always resolves to the same lockfile (including HF commit + file digests).

Runner MUST:

- fetch artifacts only via lockfile entries and verify digests before use.

### 5.2 HF caching/mirroring (recommended)

For datacenter scale, deployments SHOULD use an internal cache/mirror keyed by content digest, but the lockfile remains the source of truth for artifact digests.

## 6. Nix as the Reference Environment Builder

### 6.1 Nix closure

Environment Builder MUST produce a Nix closure that includes:

- vLLM version/commit and build inputs,
- PyTorch build,
- CUDA user-space libs (or containerized equivalent),
- kernel libraries (flash-attn/triton/xformers/etc.),
- deterministic userspace networking stack binaries + PMD/driver artifacts.

The Nix closure hash (or OCI digest derived from it) is recorded as `runtime_closure_digest`.

### 6.2 OCI distribution (optional but common)

The closure MAY be exported as an OCI image; Kubernetes workloads MUST reference images by immutable digest.

## 7. Kubernetes Execution Model

### 7.1 Workload types

- **Single-node job (current baseline)**: one Pod runs Runner + vLLM; optional networking sidecar.
- **Multi-node replicated serving**: multiple identical Pods; deterministic dispatcher routes requests.
- **Multi-node tensor/pipeline parallel**: requires pinned collective stack and additional tracing (see §10).

### 7.2 Required Kubernetes inputs

Each run MUST include:

- exact manifest copy
- exact lockfile copy
- `runtime_closure_digest`

These MUST be mounted into the Pod (e.g., ConfigMap/Secret or artifact volume) and recorded into the run bundle.

### 7.3 Hardware conformance enforcement on Kubernetes

Manifest declares hardware constraints and strictness:

- If `strict_hardware=true`, Runner MUST refuse to run on non-conforming hardware.
- If `strict_hardware=false`, Runner MAY run but MUST label results non-conformant and report diffs.

Network conformance is part of hardware conformance (NIC model/PCI ID/firmware, link settings, offloads).

Kubernetes scheduling SHOULD enforce constraints via node labels/affinity/taints; Runner MUST still validate at runtime and enforce strictness.

## 8. Batching and Engine Trace Requirements

### 8.1 Batching

Batch size is always pinned; manifest must specify cardinality constraints and policy (`fixed` preferred for strongest reproducibility).

### 8.2 Engine trace

Manifest MUST specify whether to record engine trace and which events, including:

- batch composition per step
- request reorder events
- attention backend selection
- collective algorithm selection

Runner MUST include engine trace in run bundle when enabled.

## 9. Networking Determinism (Userspace Stack + L2 Egress)

### 9.1 Contract

1. System MUST route all network I/O through a deterministic userspace networking stack.
2. Full networking stack closure MUST be pinned and recorded.
3. Egress traffic MUST be reproducible at L2 according to manifest scope and ordering rules.

### 9.2 Offloads, segmentation, and queueing

Manifest MUST explicitly define MTU/MSS, segmentation behavior (TSO/GSO policy), checksum offload policy, threading/affinity, queue mapping, ring sizes, and any internal batching; these MUST be pinned and included in runtime closure/provenance.

### 9.3 Capture without perturbation

Capturing network egress MUST not affect packetization or ordering; capture in userspace stack pre-enqueue or mirrored deterministic ring.

### 9.4 Security mode

Manifest MUST declare security mode:

- `plaintext` (recommended for strict determinism in controlled environments)
- `tls_deterministic_test_only` (explicit warnings; test only)
- `tls` allowed only if egress reproducibility is disabled

## 10. Multi-Node Scaling Requirements

### 10.1 Replicated single-node servers (recommended first)

- All Pods run identical pinned closures (same `runtime_closure_digest` + lockfile).
- A deterministic dispatcher (also pinned) controls request ordering and routing.
- Verifier compares per-Pod bundles and end-to-end outputs per manifest rules.

### 10.2 Tensor-parallel / pipeline-parallel (advanced)

Additional requirements:

- Collective stack versions/config MUST be pinned (e.g., NCCL artifacts or container digest).
- Engine trace MUST record collective algorithm selection and relevant backend decisions.
- Hardware conformance constraints likely must be stricter (topology-sensitive).

## 11. Observables and Comparison Semantics

Manifest MUST define per-observable comparison semantics: `exact`, `ulp(n)`, `absrel(atol, rtol)`, or `hash`; network egress may be compared as exact canonicalized frame bytes or hash over canonicalized PCAP stream.

## 12. Provenance and Run Bundle Format

Each run bundle MUST include:

- exact manifest and lockfile used
- runtime closure digest
- all resolved artifact digests
- environment info (vLLM/torch/CUDA metadata, GPU inventory/driver)
- execution trace metadata (actual batch sizes, resolved args/env)
- network provenance + capture metadata

Provenance MUST be sufficient for a third party to re-run and verify.

## 13. Verification Outputs

Verifier MUST produce:

- `verify_report.json` (machine readable)
- `verify_summary.txt` (human readable)

On mismatch, verifier MUST report first divergence location, numeric diff stats, batch trace diffs, network trace diffs (first diverging frame + byte offset summaries), and environment diffs (runtime closure digest, versions, hardware fingerprint).

Verifier MUST assign determinism grading: conformant / non-conformant hardware/software/network / mismatch outputs.

## 14. Kubernetes Reference Deployment Pattern

Use the following Kubernetes composition:

- **Pod**:
    - Container A: `runner` (includes vLLM invocation)
    - Container B (optional): `netstack` userspace deterministic NIC stack
    - Shared volumes for manifest+lockfile and run-bundle outputs
- **Scheduling**:
    - node selectors / affinities for GPU + NIC conformance
    - runtime validation in Runner remains authoritative for strictness
