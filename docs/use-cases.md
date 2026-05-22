# What can you do with this? — Ideas & use cases

> **In one sentence:** this project makes AI systems *reproducible and provable* —
> two different computers running the same model give the **exact same answer,
> bit for bit** — which lets you prove what an AI system actually did.

If someone asks *"what could I use this for?"*, this page is the answer. Each idea
below is written plainly first, with a pointer for engineers who want to jump
straight to the code.

---

## Why reproducibility matters (the plain version)

Normally, run an AI model twice and you can get slightly different answers — even
on the same input. That tiny wobble makes it impossible to *prove* anything: you
can't tell an honest mistake from tampering, or reproduce a result exactly.

This stack removes the wobble. Same model + same input → **identical output,
every time, on any matching machine.** Once outputs are exactly reproducible, you
can audit them, verify them, and share them.

---

## Ideas

### 🔁 "Two servers, identical answers"
**For everyone:** run your AI service on two independent machines and get
byte-for-byte identical results — proof the service is behaving consistently and
hasn't been quietly changed.
**→ Engineers:** `workflows/deterministic_inference_server.py`

### 🕵️ Catch a model doing something it shouldn't
**For everyone:** check whether an AI service that's *supposed* to just answer
questions is secretly training on your data or smuggling information out — using
only outside evidence, without trusting the operator.
**→ Engineers:** the prover-verifier demo + `workflows/verified_inference.py`

### ✅ Prove the math was actually done
**For everyone:** get a cheap, independent receipt that the heavy computation a
provider charged you for was really performed correctly.
**→ Engineers:** matmul attestation (`modules/attestation`, Freivalds' algorithm)

### 📒 Share an experiment as a single file
**For everyone:** instead of writing a page describing "here's what I ran,"
hand a colleague one short file they can run to reproduce your exact workload.
**→ Engineers:** the [recipe book](../workflows/) (`workflows/`)

### 🎯 Reproducible fine-tuning (LoRA)
**For everyone:** fine-tune a model in a way someone else can reproduce exactly,
down to the environment it ran in.
**→ Engineers:** `workflows/deterministic_lora_training.py`

### 📡 Tamper-evident network traffic
**For everyone:** make the data a server sends over the wire perfectly
predictable, so any deviation is a red flag.
**→ Engineers:** `modules/network` (deterministic egress frames)

### 🧹 Prove a machine wiped its memory
**For everyone:** get cryptographic proof that a computer actually erased
sensitive data from its memory, rather than just claiming it did.
**→ Engineers:** `modules/memory` (Proof of Secure Erasure)

### 🏗️ Reproducible builds
**For everyone:** rebuild the exact same software environment from scratch and
get an identical result — the foundation everything else rests on.
**→ Engineers:** `modules/build` (`nix build .#oci`)

---

## Try it in 30 seconds (no GPU needed)

```bash
python3 workflows/deterministic_inference_server.py
# verify status : conformant
# egress frames : 1 (reproducible: True)
```

That runs a small workload twice and confirms the two runs are byte-identical.

---

## Go deeper

- **[The recipe book](../workflows/)** — runnable, copy-pasteable workflows
- **[Capability modules](../modules/)** — the building blocks, each with a documented interface
- **[How it's organized](plans/repo-modularization.md)** — the design

## Have an idea?

These are just starting points. If you have a workload you'd like to make
reproducible or verifiable, the building blocks in [`modules/`](../modules/) are
meant to be combined — open an issue or a draft recipe and let's talk.
