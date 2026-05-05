# Prover-Verifier Demo — Memo

## Problem

A serving operator (the **prover**) runs models on hardware they control;
their customer (the **verifier**) wants on-the-fly evidence that the prover
is doing the inference workload they paid for and *only* that workload —
not training a side model on prompts, not exfiltrating data, not loading
unauthorized adapters. We want a wire-protocol-level demo that detects
each of those cheating modes from external evidence alone.

## Architecture

Two stdlib HTTP servers exchange a fixed protocol:

- `GET /graph` — the prover's claim about what it is doing (placeholder
  here; the production version threads the real task graph).
- `POST /workload/{start,stop}` — the prover starts a named workload
  (`benign`, `mixed_lora`, `lora_loading`); `/stop` returns the workload's
  internal `claimed_flops_total` + `observed_flops_total`.
- `POST /traffic` (continuous) + `POST /traffic/finalize` — the prover
  publishes a stream of bytes on-wire to the verifier; the verifier
  appends to `traffic.bin` and emits a `traffic.digest` over a running
  SHA-256.
- `POST /replay` (NDJSON streaming) — the verifier issues a Freivalds
  + PoSE-style erasure challenge; the prover streams `pow` chunks per
  matmul and a final `evidence` chunk. The verifier re-runs Freivalds on
  every per-matmul attestation pulled from `GET /attestation/{id}` and
  rebuilds the output commitment from the streamed bytes.

The verdict engine combines three signals at the verifier:

1. **replay_correctness** — every recorded `/replay/verdict/{id}` was 200.
2. **compute_budget** — `Σ observed_flops ≤ (1 + 0.10) · Σ claimed_flops`,
   summing both the scheduler's evidence and the workload's
   `/workload/stop` summary. The 10 % slack is a (weak) hand-wave for
   future kernel-timing noise; spurious for the int-counted FLOPs in this
   demo, kept on this signal for now.
3. **bandwidth** — `traffic_size ≤ claimed_artifact_bytes`, **zero
   tolerance**. The verifier observes ground-truth bytes from a trusted
   tap and the claim is an integer byte count; any extra byte is
   unattributed traffic. (See "Limitations" — the principled check is
   structural per-transmission attribution against the graph, not a
   sums-comparison; this signal is a stepping stone to that one.)

Any failed signal flips the binary verdict to `training_or_exfil` with
the failing reasons concatenated; only when all three pass does the
verdict say `inference`.

### Honest run

```mermaid
sequenceDiagram
    participant V as Verifier
    participant P as Prover
    V->>P: GET /graph
    P-->>V: 200 Graph { tasks=[...], claimed_flops=Σf }
    P->>V: POST /traffic (workload bytes, streamed)
    V->>P: POST /replay { rounds, dim, dtype=int8 }
    P-->>V: NDJSON kind=pow (per matmul)
    P-->>V: NDJSON kind=evidence
    V->>P: GET /attestation/{id} (per round)
    P-->>V: 200 Challenge+Response
    V->>V: Freivalds re-run + erasure replay → /replay/verdict 200
    V->>V: verdict = inference (all signals pass)
```

### Cheated run (mixed_lora)

```mermaid
sequenceDiagram
    participant V as Verifier
    participant P as Prover
    V->>P: GET /graph
    P-->>V: 200 Graph (placeholder: 0 tasks, 0 claimed_flops)
    P->>P: hidden gradient_steps × matmul_dim^3 work
    P->>V: POST /traffic (only the inference frames; gradient steps emit no bytes)
    V->>P: POST /workload/stop
    P-->>V: 200 { claimed_flops_total: 5120, observed_flops_total: 21504 }
    V->>V: compute_budget fails: 21504 > 1.10 · 5120 (compute keeps 10 % slack)
    V->>V: verdict = training_or_exfil
```

The `lora_loading` scenario is symmetric on the bandwidth axis: the
workload downloads a 512 KB blob from an external URL and republishes
the bytes on-wire **without** calling `record_task`, so `traffic_size`
exceeds `claimed_artifact_bytes` (by the size of the unattributed blob)
and the bandwidth signal trips.

## Threat model

**Detected.** A prover that runs more compute than it claims (training,
hidden adapters), or one that ships more bytes on-wire than the inference
graph accounts for (model exfiltration, side-channel data leakage, LoRA
loading from unauthorized URLs).

**Not detected here.** A prover that hides cheating *inside* a single
recorded task (e.g. larger inference batches than declared), a colluding
multi-prover setup that splits work across endpoints we don't monitor,
or pre-arranged bit-perfect collusion with the workload-level
`claimed_flops`. The placeholder graph means we can't yet verify
*which* tokens were inferred — only that the claimed/observed totals add
up. Phase 11 will replace the placeholder with a real task graph and
swap the bandwidth baseline for a per-task budget.

## Results

The smoke sweep covers one knob value per workload:

- `benign` (seed=0) → `inference` ✓
- `mixed_lora` (gradient_steps=8) → `training_or_exfil` ✓ (compute_budget
  fail: observed=21504 vs claimed=5120; ratio ≈ 4.2×)
- `lora_loading` (lora_bytes=524288) → `training_or_exfil` ✓ (bandwidth
  fail: traffic=526848 vs claimed=2560; ratio ≈ 205×)

Detection curves over the full knob sweep:

![mixed_lora detection](../figures/mixed_lora_detection.png)

![lora_loading detection](../figures/lora_loading_detection.png)

Per-row inspection lives in the [self-contained HTML
viewer](../viewer.html) (`file://` — no server needed).

## Failure modes / limitations

- **No TLS, no authentication** between prover and verifier — assumed
  out-of-scope for the demo. A production deployment would tunnel
  `/replay` and `/traffic` over an attested TLS link with a verifier-pinned
  cert.
- **Placeholder graph.** `GET /graph` returns an empty `tasks=[]` body.
  The compute-budget signal works around this by reading the workload's
  internal `observed_flops_total` from `/workload/stop`; a real
  deployment would derive both totals from the task graph.
- **Single pod.** All workloads run in one prover process; no
  cross-pod scheduling or task graph dependencies.
- **Stdlib only.** No DPDK, no kernel-bypass, no GPU path. `enforce_eager`
  + Freivalds at int8 is enough on CPU; `bf16`/`fp16` paths are gated on
  a torch backend that lights up only on GPU hosts.
- **Compute-budget tolerance is fixed at 10 %.** Picked to swallow
  timestamp noise and benign drift; an adversary that stays under 10 %
  over-budget evades the compute signal. The full sweep would tune this
  per-workload. (Bandwidth's tolerance, by contrast, is **zero** — bytes
  are bytes; see next bullet.)
- **Bandwidth signal is sums-comparison, not structural attribution.**
  The principled check is: every observed transmission's
  `{tap_id, tap_seq, artifact_id, commitment, size}` matches a transmission
  declared in the graph; anything observed-but-not-declared is the cheating
  signal, and the predicate has no slack at all. The current implementation
  compares totals because the placeholder `/graph` declares no transmissions
  to attribute against, and `/traffic` frames are untagged byte streams. The
  totals check is enough for `lora_loading` (a 200× blow-up isn't subtle)
  but it would miss an attacker who added a few unattributed bytes between
  every legitimate frame. Real fix lands with the task graph.

## Reproducing on two machines

The demo's `--remote` mode skips the local spawn and trusts that prover
and verifier are already running on whatever hosts you point it at:

```bash
# On the prover host (e.g. Lambda gpu_1x_h100):
python3 cmd/prover/main.py --host 0.0.0.0 --port 8000 \
    --run-id prod-001 --out-dir /tmp/prover \
    --verifier-url http://10.0.0.2:9000

# On the verifier host:
python3 cmd/verifier_server/main.py --host 0.0.0.0 --port 9000 \
    --out-dir /tmp/verifier \
    --prover-base-url http://10.0.0.1:8000

# On either machine (or a third):
PROVER_HOST=10.0.0.1 PROVER_PORT=8000 \
VERIFIER_HOST=10.0.0.2 VERIFIER_PORT=9000 \
  ./experiments/prover-verifier-demo/demo.sh --remote
```

The full provisioning + networking recipe (Lambda Cloud, vast.ai,
Digital Ocean GPU droplets, key setup, port allow-lists) is in the
"Cloud GPU provisioning" section of
[docs/plans/prover-verifier-demo.md](../../../docs/plans/prover-verifier-demo.md).

## Next steps

1. Replace the placeholder graph with the real task graph from
   `experiments/task-graph-prototype/` so the compute and bandwidth
   baselines come from declared per-task budgets, not workload self-report.
2. End-to-end encryption (TLS or Noise) on `/replay` and `/traffic`
   with verifier-pinned certs.
3. Adversarial robustness sweeps: how many gradient steps can hide under
   `tolerance`? Tune the tolerance against benign timing variance from a
   real workload.

---

*Reproduce locally: `cd experiments/prover-verifier-demo && ./demo.sh
--quick` exits 0 with `ALL PASS` (~15 s wallclock).*
