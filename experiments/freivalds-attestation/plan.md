# Freivalds Matmul Attestation

**Goal.** Add a probabilistic compute-attestation primitive to the deterministic serving stack: a verifier (orchestrator) issues randomized matmul challenges to a prover (the deterministic serving server, or a stand-in for it), the prover executes the matmuls on the GPU and returns the results, and the verifier uses Freivalds' algorithm to confirm the answers cheaply (O(n²) per matmul instead of O(n³)).

The attestation power comes from two structural properties:

1. **Saturation.** A large square GEMM hits ~75–90% of peak tensor-core FLOP/s on H100 SXM5 (e.g. ~750 of ~989 TF/s bf16). While the kernel runs on a single CUDA stream with no MPS/MIG, other work cannot run on those SMs — the GPU genuinely cannot be doing two things at once.
2. **Predictable wall-clock.** For dim ≥ 4096 with bf16/fp16 the kernel runs long enough that launch and tail effects are negligible; observed time on a thermally stable machine has only a few percent variance.

Combined: a verifier that samples honest runs to learn `T_expected(dims, dtype, hw)` can later use observed-time as a gating signal in v2. v1 measures and records `T_observed` but does not gate on it.

---

## Threat model

The prover may be:

- **honest** — establishes the calibration baseline;
- **returning cached outputs** from a precomputed table (defeated by fresh seeds + dim/dtype sweep);
- **returning random/zero matrices** (Freivalds detects);
- **running a stub kernel** that does no real GPU work (Freivalds detects, and v2 timing will);
- **(v2 only)** delegating to a faster GPU than claimed (timing-based gating; out of v1 scope).

**Out of scope for v1.**

- fp4 (no native H100/GH200 support; revisit on Blackwell).
- Multi-GPU matmul attestation (single GPU only).
- Cryptographic / zero-knowledge commitments over `A`, `B`, `C`.
- Manifest/lockfile integration. The verifier passes parameters per-call via the request body. Promotion to a manifest-declared challenge profile is a follow-up once the protocol is validated.
- Confidential Compute (CC) mode (counter telemetry disabled — same reason as `experiments/flop-attestation`).

---

## Soundness

Textbook Freivalds: pick uniform `r ∈ F^N`; `Pr[A·(B·r) = C·r | A·B ≠ C] ≤ 1/|F|`. Repeat with `k` independent `r` to drive failure probability to `(1/|F|)^k`.

Under float arithmetic with tolerance ε, soundness becomes:

> *any divergence `‖AB − C‖ > ε_calibrated` is detected with probability ≥ 1 − δ over `r`.*

The attestation is sound against a malicious prover because:

1. **`r` is generated and consumed entirely on the verifier.** The prover never sees `r` and so cannot engineer `A(Br) = Cr` without producing `C`.
2. **Seeds for `A` and `B` are fresh per challenge** and revealed only at challenge time. Across the dim/dtype sweep, lookup-table attacks are impractical.
3. **`r` is generated *after* the prover has returned `C`.** Since `C` is committed (sent in full), the prover cannot change it in response to `r`.

Two verification regimes:

| Mode | Inputs | Comparison |
|---|---|---|
| **bitwise** | int8 inputs, int32 accumulator | `A(Br) == Cr` exactly |
| **tolerance** | bf16/fp16/fp8/fp32 inputs, fp32 accumulator | `‖A(Br) − Cr‖∞ ≤ atol + rtol · ‖Cr‖∞` with calibrated `atol`, `rtol` |

The bitwise mode gives a clean soundness statement and runs on every supported integer dtype. The tolerance mode is what runs on the inference dtypes the project actually uses — float ε is calibrated empirically from honest runs (see Phase 2).

---

## Protocol

Single-round, prover returns full `C`.

```
verifier                                                            prover
   │                                                                  │
   │  POST /attest  { challenge_id, matmuls: [{seed_a, seed_b, …}] } │
   ├─────────────────────────────────────────────────────────────────>│
   │                                                                  │  for each matmul:
   │                                                                  │    A = prng(seed_a, dtype_a, M, K)
   │                                                                  │    B = prng(seed_b, dtype_b, K, N)
   │                                                                  │    C = matmul(A, B, dtype_acc, dtype_c)
   │                                                                  │    record wall_time, nvml clock/temp
   │  200 OK   { results: [{ digest_a, digest_b, c_bytes, timing }] } │
   │<─────────────────────────────────────────────────────────────────┤
   │                                                                  │
   │  for each matmul:                                                │
   │    A = prng(seed_a, …); B = prng(seed_b, …)                      │
   │    assert digest_a, digest_b match (PRNG drift check)            │
   │    pick r locally from a fresh random source (not from the wire) │
   │    Br = B·r;  ABr = A·(Br);  Cr = C·r                             │
   │    verdict = bitwise/tolerance(ABr, Cr)                           │
   │  emit attestation report (verdict per matmul + timing)            │
```

Notes:

- **`r` never leaves the verifier.** This is the structural property that makes the protocol attestation-sound.
- **`C` is delivered as raw bytes** (with the dtype + shape declared in the JSON envelope). For 4096×4096 bf16 that is 32 MiB per matmul; the wire format wraps the JSON envelope and a binary blob per matmul (multipart-like, but represented as a base64 byte field in the JSON body in v1 to keep the experiment plain — performance is not a v1 goal).
- **`digest_a` and `digest_b` are SHA-256 of canonical row-major bytes** at the declared dtype. The verifier re-runs the PRNG locally and compares; mismatch = PRNG drift between sides, not a Freivalds failure. This separates "your random number generator is wrong" from "your matmul is wrong" early.
- **Verifier picks parameters per-call.** No manifest pre-commit in v1. Different probes (small-int sanity, large-bf16 saturating, mixed-dtype sweep) are different request bodies.
- **Timing is observed and recorded but does not gate.** v1 deliverable: a calibration dataset of honest runs at a sweep of (dim, dtype). v2 turns this into a `T_expected(dims, dtype, hw)` curve and a gating threshold.

---

## Codebase layout

```
pkg/freivalds/
  __init__.py                public API
  spec.py                    Challenge / MatmulSpec / Response dataclasses + JSON serde
  prng.py                    seed → matrix bytes (pluggable backend)
  check.py                   bitwise + tolerance Freivalds check
  prover.py                  execute(challenge) → response (uses backend)
  verifier.py                verify(challenge, response) → AttestationReport
  backends/
    __init__.py              backend registry
    stdlib.py                pure Python (float64, int8, int32) — for tests + CPU fallback
    torch_backend.py         torch tensors (all GPU dtypes) — used in production

schemas/
  freivalds_challenge.v1.schema.json
  freivalds_attestation.v1.schema.json

experiments/freivalds-attestation/
  plan.md                    this file
  EXPERIMENT_LOG.md          append-only log
  scripts/
    run_smoke.py             in-process honest+adversarial round-trip on CPU
    serve.py                 (Phase 2) HTTP prover server
    challenge.py             (Phase 2) HTTP verifier CLI
    calibrate.py             (Phase 2) sweep honest runs to learn T_expected and ε
    probe_cached.py          (Phase 3) adversarial: prover returns cached C
    probe_zeros.py           (Phase 3) adversarial: prover returns zero C
    probe_partial.py         (Phase 3) adversarial: prover skips layers / drops dims
  data/                      challenge bundles, calibration grids
  reports/                   write-ups
  figures/                   plots

tests/unit/
  test_freivalds_prng.py     stdlib PRNG determinism + dtype quantization
  test_freivalds_check.py    Freivalds detects honest vs corrupted answers
  test_freivalds_protocol.py prover/verifier round-trip on small int + float
  test_freivalds_schemas.py  schema validates / rejects
```

---

## Calibration story (Phase 2)

For each `(dim, dtype, hw)` triple in the sweep:

1. Run `n_trials=20` honest matmul challenges back-to-back on a representative H100 SXM5 (or GH200) with the deterministic stack environment vars.
2. Record per-trial: `wall_time_ms`, `‖A(Br) − Cr‖∞`, `‖Cr‖∞`, NVML clock MHz, NVML temp.
3. Fit:
   - `T_expected = median(wall_time_ms)` with IQR for variance bars.
   - `ε_atol`, `ε_rtol` such that 99.9% of honest runs satisfy `‖ABr − Cr‖∞ ≤ atol + rtol · ‖Cr‖∞`. Choose the *minimum* viable tolerance — larger ε weakens soundness against tolerance-aware attacks.
4. Cross-validate by running another `n_holdout=20` and confirming the calibrated bounds hold.

Sweep:

- Square dims `{1024, 2048, 4096, 8192, 16384}` (largest within memory).
- Decode-shape `{(1, 4096, 4096), (32, 4096, 11008)}` for dtype coverage but flagged as non-saturating.
- Dtypes: `int8→int32`, `bf16→fp32→bf16`, `fp16→fp32→fp16`, `fp8_e4m3→fp32→bf16` (subject to torch/cuBLAS support on the test hardware).

Output: `experiments/freivalds-attestation/data/calibration_v1.json` keyed by `(hw_fingerprint, dim, dtype)`.

---

## Adversarial probe matrix (Phase 3)

| Probe | What the prover does | Expected verdict |
|---|---|---|
| **S0 honest** | Real GPU matmul, returns true C | `verdict=pass`; timing within calibrated band |
| **S1 cached** | Returns precomputed C from a cache (correct for previously-seen seed; wrong for fresh seed) | `verdict=fail` on fresh seeds (Freivalds catches); `pass` if seeds reused — establishes that fresh seeds matter |
| **S2 zeros** | Returns C = 0 | `verdict=fail` |
| **S3 random** | Returns C = random matrix | `verdict=fail` |
| **S4 dropped-rows** | Computes C correctly for the first half of rows, zeros the rest | `verdict=fail`; localized divergence |
| **S5 quantized-cheat** | Casts B to int4, multiplies, casts back, claims fp16 | `verdict=fail` if calibrated ε is tight enough; this is the test of how tight ε is |
| **S6 stub-kernel** | Launches a busy-loop kernel, returns garbage C | `verdict=fail`; v2 timing will look honest, Freivalds catches the answer |

For each: report `(verdict, ‖ABr − Cr‖∞, observed_time_ms)`. Detection margin = the gap between the honest band and the probe.

---

## Phasing

| Phase | Scope | Exit |
|---|---|---|
| **P1 — Protocol & stdlib path (this PR)** | spec, prng, check, prover, verifier on stdlib backend; schemas; unit tests; in-process smoke | All unit tests pass on a CPU-only box; smoke shows honest=pass, zeros/random=fail |
| **P2 — Torch backend & calibration** | torch backend for bf16/fp16/fp8/int8; HTTP server + CLI; calibration sweep on H100 | `data/calibration_v1.json` exists; honest-run band measured |
| **P3 — Adversarial probes** | S1–S6 probes wired up; detection-margin report | Probe matrix table in `reports/p3_detection_margin.md` |
| **P4 — Promote endpoint** | `POST /attest` on `cmd/server/main.py`; `cmd/freivalds_verifier/`; manifest challenge profile | Endpoint live; integration test in `tests/integration/` |

---

## Non-goals

- Not a zero-knowledge proof. Hypervisor / network-level trust is assumed (same as `experiments/flop-attestation`).
- Not a defense against firmware/driver compromise.
- Not cross-vendor (NVIDIA only in v1).
- Not a primitive for proving that an *inference* — as opposed to a synthetic matmul — used real GPU work. Promoting from synthetic to "sample a real matmul from a transformer layer and challenge it" is a follow-up that requires manifest integration and is captured in P4 only as a stub.
