# Streaming/strided proof-of-compute protocol — spec v0

**Status:** draft. Designed by Luke and Buck on 2026-04-30, written up to satisfy Buck's "GBT 5.5 Pro just write this up as a spec" line. This document supersedes the single-shot Freivalds protocol described in `plan.md` for use cases where the verifier needs a refresh-rate guarantee on prover bandwidth.

## Why the v1 single-shot protocol doesn't suffice

In the v1 protocol the prover receives `K` matmul specs, computes them all, and returns a single response containing `K` C-bytes payloads plus their digests. The verifier runs Freivalds locally on each C.

The problem (Buck, 2026-04-30):

> If you tell the prover, do a thousand matmuls, and then after you've done all of those, send me the results of those matmuls — versus if you tell the prover, do a matmul, and then send me the results of that matmul, and you tell them to do that a thousand times — they get to exploit whatever connections they have to other computers way more in the first setting. Because imagine they have a one gigabit connection to another rack: they can use that to get the other rack to do way more matmuls if you just tell them to tell you the result at the end.

Until the prover's first response, an adversarial prover can offload the full work over an external network at the bandwidth of that link. The single-shot protocol therefore only bounds total compute, not the **rate** at which the prover must produce that compute on its own hardware.

## The protocol

### Roles
- **Verifier** (V): the auditor. Operates a high-bandwidth, low-latency channel to the prover. Knows total work `K · 2 · n³` FLOPs.
- **Prover** (P): the GPU under audit. Wants to convince V that all FLOPs happened on this physical device.

### Parameters (chosen by V before challenge)

| Symbol | Meaning |
| --- | --- |
| `K` | Total number of matmuls in the challenge. Multiplied with `2·n³` gives the FLOP budget V is paying for. |
| `M` | Stride. Number of matmuls per response. `1 ≤ M ≤ K`. **Security parameter.** Smaller M = more frequent rounds = tighter bandwidth bound. |
| `n` | Width of each square matmul. `n × n × n` per matmul. **Per-matmul compute parameter.** |
| `s` | Seed size in bits. **Bandwidth-asymmetry parameter** — must be small enough that streaming a seed is much cheaper than streaming `n²` matrix elements, but large enough that an adversary can't precompute matrices for all seeds. 64 bits (default) gives a $2^{64}$ search space. |
| `h` | Hash output digest length. Defaults to 256 bits (sha256). |

Hard constraints:
- The matrix bandwidth between V and P must exceed the bandwidth between P and any external computer P could solicit help from. (Standard secure-erasure assumption.)
- One round of `M` matmuls should take ≈ 1 s of P's wall clock. P's matmul rate is ~10 µs at `n = 1024`, so M ~ 10⁵ at that size; M ~ 10³ at `n = 4096`. Don't drive the round time below ~1 s — communication cost dominates and the channel can't keep up.

### Wire format

Each chunk of M seeds is a single message from V to P:
```
ChunkRequest {
  chunk_index: int
  seeds: [(seed_a, seed_b)] × M
  matmul_ids: [str] × M
}
```

Each chunk response is a single message from P to V:
```
ChunkResponse {
  chunk_index: int
  matmul_ids: [str] × M
  chain_hash: bytes(h)
  wall_time_ms: float
}
```

`chain_hash` is computed by P as follows. Let `digest_c_i = sha256(canonical_bytes(C_i))` for matmul `i` in the chunk. Define the genesis `h_0 = sha256("freivalds-streaming-chain-v1|genesis")`. Then:

```
h_i = sha256(h_{i-1} || "|" || digest_c_i)     for i = 1..M
chain_hash := h_M
```

The exact byte layout (single-byte separator, ASCII hex prefixes) is documented in `pkg/freivalds/streaming.py::fold_chain_hash`. Order matters: the chain depends on the position of each matmul in the chunk, so the prover cannot reorder or substitute.

### Per-chunk matmul construction

For each `(seed_a, seed_b)` in a chunk, P:

1. Expands `seed_a` → canonical bytes for an `n × n` matrix `A` of declared dtype, using the existing `pkg.freivalds.prng` SHAKE-based scheme.
2. Expands `seed_b` → canonical bytes for an `n × n` matrix `B` of declared dtype.
3. Computes `C = A · B` on its own GPU (no migration off the device).
4. Hashes the canonical row-major bytes of `C` to get `digest_c`.
5. Folds `digest_c` into the running chain hash.

After all M matmuls in the chunk are processed, P sends `ChunkResponse` and waits for V's next request.

### Hard constraints on seeds and matrices

> "I would be surprised if [multiplying the same matrix by itself] didn't lead to some fuckery. I think they should definitely just be two random matrices, two different random matrices." — Buck

> "We don't want to seed the matrices using low-rank noise. But I agree we can generate the elements using just a seed." — Buck

Therefore:
- **Two distinct seeds** per matmul. `seed_a ≠ seed_b`. Never `A · A`.
- **Full-rank matrices.** The PRNG produces uniform-bit-pattern entries (post-twiddling) so the matrices are dense and full rank with overwhelming probability. Never seed by combining low-rank noise with a known basis.

### Verifier behaviour

V issues chunks back-to-back, no gap, and times each round trip. Chunks are accepted only if:

1. `chain_hash` matches V's local recomputation. (V re-derives `(A, B)` from the seeds it sent and re-runs the same matmul on its own hardware — or on a separate honest prover.)
2. The round-trip time is within an upper bound `T_max`. `T_max` is set to roughly `M × t_matmul_honest + slack`, where `t_matmul_honest` is the calibrated per-matmul time on the prover's GPU class and `slack` covers network jitter.

In production, V doesn't recompute every chunk — it samples. This module's `verify_streaming_response` recomputes all chunks; sampling is a config knob to be added later.

### What this protocol proves

If V's bandwidth assumption holds and all chunks pass within `T_max`, then with overwhelming probability the prover ran `K · 2 · n³` FLOPs on its own hardware during the challenge window. More precisely:

- **Compute lower bound.** Every chunk requires `M · 2 · n³` FLOPs of dependent work (the chain hash forces serialisation across the M matmuls in the chunk).
- **Bandwidth upper bound.** Within one round of duration `~M · t_matmul_honest`, P's external link can push at most `B_ext · M · t_matmul_honest` bits. For honest values of `n` (≥ 1024) and the matrix dtype, `n²` floats already exceeds this — so P cannot offload even a single matmul's full input over the external link in a single round.
- **No precomputation.** Seeds are unpredictable to P (V draws them from its own RNG), so P cannot precompute results.

The proof is information-theoretic in the bandwidth assumption + computationally bound by sha256.

### Out of scope (for this spec)

- Sampling strategy for V.
- Mixed-precision matmuls (Luke noted modern models use these; the protocol is dtype-agnostic but hasn't been calibrated for mixed pipes yet).
- Zero-knowledge wrapping. The IEEE S&P paper aim is to make the overall protocol zero-knowledge; this chain-hash construction does not yet hide `digest_c` from V (V learns it during recomputation).

### Implementation pointers

- `pkg/freivalds/streaming.py` — `execute_streaming_challenge`, `verify_streaming_response`, `fold_chain_hash`.
- `pkg/freivalds/spec.py` — `Challenge.matmuls_per_response`, `Response.chain_hashes`, `ChainHashChunk`.
- `experiments/freivalds-attestation/scripts/sm_occupancy_sweep.py` — `OccupancyController.occupy_flops(flops, duration_s)` is the FLOPs-native interface; the challenge function now takes a FLOP budget and converts to `(K, n, M)` internally.

### Origin

Captured from the meeting transcript Buck → Luke → Jonathan, 2026-04-30:

> "The replay server uses [the seed] to populate two n by n matrices and then multiply them together and then hash the result. And then for each of these M random matmuls it just takes a hash chain of each of their outputs and then responds with that. And then we just repeat this K/M times."
