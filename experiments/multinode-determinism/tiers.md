# D6 Determinism Tiers — Smoke / Medium / Large

Companion to `docs/plans/d6-lambda-staged-rollout.md`. Specifies a tiered set
of determinism experiments that scale the inference workload from a few tokens
(seconds, dollars) to ~1M tokens (hours, ~$80) so we can buy increasing
confidence in the D6 claim per dollar spent.

## What "determinism" means here

For each tier, we run the **same manifest + lockfile + cluster** twice (or
three times for a stronger signal) and assert that the **`request_outputs[].tokens`** arrays are
**bitwise identical**, request-by-request. Decoded text is not compared —
tokens are the canonical signal. Logit comparison is skipped per-tier
(separate concern).

Two derived properties:

1. **Same-config repeat (A == A')** — the model is deterministic in the
   trivial sense: rerun the same inputs and get the same outputs.
2. **Batch-order invariance (A == B-shuffled, by request id)** — shuffle the
   prompt order within the manifest (which changes batch boundaries, padding,
   prefill order, MoE expert routing order). Tokens per request must be
   identical to A. This is the harder test and the one D6 cares about.

Tier 1 only checks (1). Tier 2 checks (1) + (2). Tier 3 checks (1) only,
because at 1M tokens the cost of running a third shuffled pass is non-trivial
and we already have stronger statistical confidence from the volume.

## Topology assumed

- 4× H100 SXM5, same Lambda region (us-south-2), public-IP Ray cluster
- vLLM 0.17.1 + Ray 2.54.0, deterministic NCCL pinning + `sitecustomize.py`
  gloo bind patch
- `enforce_eager=True`, `temperature=0` (greedy), `seed=42`, batch invariance enabled
- TP=4 (one rank per node)

## Models

| Model | Type | Params | Size bf16 | HF repo |
|---|---|---|---|---|
| **DBRX-Instruct** | MoE, 16 experts top-4 | 132B (~36B active) | ~263 GB | `alpindale/dbrx-instruct` (mirror; original `databricks/dbrx-instruct` was pulled from HF) |
| **Mistral Large 2** | Dense | ~123B | ~240 GB | `mistralai/Mistral-Large-Instruct-2407` (gated; access granted) |

DBRX gives MoE expert-routing coverage. Mistral Large 2 gives a dense baseline
of comparable parameter count. Both are larger than the aggregate VRAM of
3× H100 (240 GB) and demand all 4 GPUs at TP=4, so completing inference is
itself structural proof of 4-way distribution.

PP=4 is **not** in scope — vLLM 0.17.1 has a separate bug
(`KeyError: transformer.blocks.10.norm_attn_norm.attn.attn`) in its attention
backend resolver when DBRX is split by layer ranges across pipeline ranks.
Filed as a known-issue in the experiment log; needs a vLLM patch or version
bump. TP=4 sidesteps it.

## Tier 1 — Smoke (both models)

**Purpose:** prove the harness runs end-to-end without any of the multi-node
plumbing dropping a token. Validates the cluster after each restart and runs
in seconds, so it's the iteration unit when something looks off.

| Parameter | Value |
|---|---|
| Models | DBRX **and** Mistral Large 2 |
| Prompts | 4 (curated, varied length) |
| `max_new_tokens` per request | 16 |
| Total tokens per run | 64 |
| Runs per model | 2 (A, A') |
| Comparison | A == A', token-exact, per model |
| Wall time per model | ~10 min (engine init × 2 + tiny inference) |
| Total wall time | ~20 min |
| Approx. cost @ $17.16/hr | ~$6 |
| Pass criterion | A.tokens == A'.tokens for all 4 requests, in each model |

Manifests:
- `manifests/dbrx-tp4-smoke.manifest.json`
- `manifests/mistral-large2-tp4-smoke.manifest.json`

## Tier 2 — Medium ~10K tokens (both models)

**Purpose:** real determinism signal across 100+ generation steps per request,
with enough varied prompts that MoE expert routing exercises many experts and
attention has heterogeneous KV cache layouts. Includes the batch-order
invariance test, which is the single most likely thing to surface a hidden
nondeterminism.

| Parameter | Value |
|---|---|
| Models | DBRX **and** Mistral Large 2 |
| Prompts | 100 (varied length 8–80 tokens, varied topics, varied tail-token sensitivity) |
| `max_new_tokens` per request | 100 |
| Total tokens per run | 10,000 |
| Runs per model | 3 (A, A', B-shuffled) |
| Comparison | A == A' (per request); A == B-shuffled (per request id, after re-sort) |
| Wall time per model | ~30 min (3 × init + 3 × inference) |
| Total wall time | ~1 hour |
| Approx. cost @ $17.16/hr | ~$20 |
| Pass criterion | All three runs token-equal per request id, in each model |

Manifests:
- `manifests/dbrx-tp4-medium.manifest.json`
- `manifests/mistral-large2-tp4-medium.manifest.json`

## Tier 3 — Large ~1M tokens (DBRX only)

**Purpose:** scale validation. If determinism holds across 100 prompts × 100
tokens, the conditional probability that it breaks at 250 prompts × 4000
tokens is small — but not zero. This tier exists to reject the failure mode
"deterministic at small scale, drifts at long context due to KV cache
accumulation rounding."

DBRX is chosen for the large tier because MoE routing is the single most
likely place for accumulated nondeterminism to bleed in over 4000 generation
steps — much more than for a dense model.

| Parameter | Value |
|---|---|
| Model | DBRX (only) |
| Prompts | 250 (a superset of the medium tier prompts, plus longer ones to reach 4096-context) |
| `max_new_tokens` per request | 4000 (just under the manifest's `max_model_len=4096`) |
| Total tokens per run | 1,000,000 |
| Runs | 2 (A, A') |
| Comparison | A == A' per request id |
| Wall time per run | ~5 min init + ~80–140 min generation (depends on `max_num_seqs=64` batching, 4 waves; throughput ≈ 100–200 tok/s aggregate with `enforce_eager` + batch invariance) |
| Total wall time | ~3–5 hours |
| Approx. cost @ $17.16/hr | ~$50–85 |
| Pass criterion | A.tokens == A'.tokens for all 250 requests |

Manifest: `manifests/dbrx-tp4-large.manifest.json`.

**Why no shuffle for Tier 3:** triple-running this tier would cost $80–130 and
add 3+ hours. The shuffled invariance test in Tier 2 already exercises batch
boundary effects on the same model. Tier 3's contribution is *length*, not
*ordering*. If you want maximum confidence, accept the extra spend and run
B-shuffled for Tier 3 too.

## Comparison rules

For each tier, the comparison loads `out/<run>/observables/tokens.json`
which is a list of `{id, tokens}`. The check:

1. Build a dict by `id` for each run.
2. Assert `set(ids_A) == set(ids_A')` (no missing requests).
3. For each id, assert `tokens_A[id] == tokens_A'[id]` (Python list equality).
4. Report first divergence: which request id, which token index, the byte at A
   vs A'. The divergence point is more informative than just pass/fail.

This logic lives in `scripts/d6/compare_observables.py` (to be added).

## Manifest generation

A small Python script `scripts/d6/generate_tier_manifests.py` builds the three
tier manifests from a single base file (`manifests/dbrx-tp4-multinode.manifest.json`)
plus an inline prompt corpus. Each generated manifest gets a fresh
`run_id` and is checked into git. The lockfile for each is generated by
`modules/inference/resolver/main.py` — same artifact list as the base since the model and
revision are unchanged, so the only diff is `manifest_digest`.

Prompt corpus design:

- **Tier 1 prompts (4):** literal copies of 4 prompts from the medium set,
  selected for short prompts that complete quickly.
- **Tier 2 prompts (100):** balanced across topic categories (science, history,
  literature, math, daily life, code, instructions, descriptive). Length
  distribution: 25% under 16 tokens, 50% 16–48 tokens, 25% 48–80 tokens.
- **Tier 3 prompts (250):** Tier 2's 100 + 150 additional. New ones include
  some that intentionally push the model into longer outputs ("explain X in
  detail", "write a short story about Y", "step by step ...").

All prompts are committed alongside the manifest so the experiment is
reproducible from the branch alone.

## Execution order

1. Generate manifests for all 6 (3 tiers × 2 models, except Tier 3 is DBRX only) + commit.
2. Push manifests to all 4 nodes (we already have a 4-node cluster up; DBRX
   weights cached).
3. Start Mistral Large 2 download to all 4 nodes (background).
4. Resolve DBRX lockfiles for all 3 tiers inside the ray-head container.
5. **Tier 1 DBRX**: A → A' → compare. Pull observables back to laptop, git commit.
6. (Wait for Mistral download.) **Tier 1 Mistral**: A → A' → compare. Pull, commit.
7. **Tier 2 DBRX**: A → A' → B-shuffled → compare. Pull, commit.
8. **Tier 2 Mistral**: A → A' → B-shuffled → compare. Pull, commit.
9. **Tier 3 DBRX**: A → A' → compare. Pull, commit. (~3–5 h)
10. Write the final report. Terminate.

After **every** run completes, observables are scped back to
`experiments/multinode_determinism/<date>/<tier>/<model>/<run>/` and committed
immediately. If anything dies mid-experiment, work is preserved.

Stop on first mismatch — don't burn budget on higher tiers if a lower tier
has uncovered a real determinism bug.

Total expected wall time: **5–7 hours**.
Total expected spend: **$80–120**.

## Failure modes and what they tell us

| Fails at | Likely cause |
|---|---|
| Tier 1 A vs A' | NCCL not pinned, env var dropped between runs, container drift, or a real new bug. Single-host bug — investigate immediately. |
| Tier 2 A vs A' but Tier 1 was clean | Larger batch surface exposes a kernel choice that wasn't deterministic — likely an autotuned reduction or attention path. Check FLASH_ATTN backend selection consistency. |
| Tier 2 A vs B-shuffled | Batch order matters for at least one prompt — order-dependent reduction in MoE routing or attention. **Real D6 finding** — log it, do not paper over. |
| Tier 3 A vs A' but Tier 2 was clean | Long-context path diverges. Possible KV cache prefix-cache nondeterminism, or accumulated rounding crossing a quantization boundary. Find the diverging token index. |

## Anti-cheat

The Phase 2 Anti-cheat checks (per-rank GPU memory, NCCL ring confirmed in
logs, iptables interdict) are NOT repeated here — they were validated on the
2-node cluster and the 4-node cluster has the same pinning + topology, just
more nodes. The structural anti-cheat is the model size: DBRX is 263 GB and
cannot fit on fewer than 4 H100s, so the mere fact that inference completes
is proof of 4-way distribution.

## What this plan does NOT cover

- PP=4 (blocked on vLLM 0.17.1 DBRX layer-resolution bug)
- Mistral Large 2 (token works now, but adds another 240 GB download per node
  and would balloon Tier 3 wall time)
- Multi-replica cross-host comparison (different physical machines running
  identical configs) — relevant for D6 but a separate axis from "same
  cluster, multiple runs"
- Logit-level comparison (we compare token IDs, not logits, because the
  observable schema's logit comparison is intentionally fuzzy and the harder
  test is exact token match)
