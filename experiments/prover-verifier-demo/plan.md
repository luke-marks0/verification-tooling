# Prover ↔ Verifier E2E Demo

**Goal.** End-to-end demo of a prover–verifier protocol that detects training (or
exfiltration) hidden inside an inference workload. The prover is a deterministic
serving server; the verifier issues graph queries, replay challenges, and
ingests a streamed traffic capture. Three workloads (one benign, two
adversarial) feed a verdict engine that flags training/exfil from honest
inference.

**Status.** in progress

**Where the implementation plan lives.** See
[`docs/plans/prover-verifier-demo.md`](../../docs/plans/prover-verifier-demo.md).
That document is the source of truth — phases, tasks, conventions, what's
in/out of scope. This file is the experiment-side mirror; keep it short.

## Scope (in)

- Two stdlib-HTTP Python servers, runnable on one box (two ports) or two boxes.
- Placeholder graph endpoint (empty-but-typed). Real task-graph integration is
  a follow-up.
- Replay endpoint with PoSE-style erasure proof + Freivalds-style PoW evidence.
- Plaintext traffic streaming. TLS deferred.
- Three workloads: benign inference, mixed inference + LoRA training (size knob),
  inference using externally-loaded LoRAs (bandwidth knob).
- Detection logic that converts transcript+traffic into a verdict.
- One evaluation plot per knob (FLOPs vs detection-rate, bytes vs detection-rate).

## Scope (out)

- Encryption/TLS. `--security-mode` flag is stubbed; only `plaintext` works.
- Multi-pod orchestration, full task-graph integration, ZK proofs.
- Real-world LoRA training (we simulate matmuls; we don't train a model that
  improves).

## Headline deliverable

`./experiments/prover-verifier-demo/demo.sh --quick` exits 0 and prints
`ALL PASS` plus a summary table showing the three verdicts match
expectations (benign → inference; mixed_lora and lora_loading →
training_or_exfil).

## Layout

```
experiments/prover-verifier-demo/
  plan.md                   this file
  EXPERIMENT_LOG.md         append-only log
  demo.sh                   headline deliverable (Phase 10)
  scripts/
    demo_driver.py          two-server orchestration for demo.sh
    run_eval.py             sweep harness (Phase 9)
    plot_results.py         matplotlib plots
    make_viewer.py          HTML viewer generator
    workloads/              one file per workload
  data/                     generated transcripts, traffic captures
  reports/                  memo
  figures/                  PNG plots
```

## Reading order for new contributors

1. `docs/plans/prover-verifier-demo.md` — full plan.
2. `EXPERIMENT_LOG.md` — what shipped and what surprised.
3. `modules/attestation/prover/main.py`, `modules/attestation/verifier_server/main.py`, `modules/attestation/verifier_cli/main.py`.
4. `pkg/proverdet/*`.
5. `demo.sh` and `scripts/demo_driver.py`.
