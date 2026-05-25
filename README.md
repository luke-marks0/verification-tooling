# Deterministic Serving Stack

Bitwise identical LLM inference across independent servers. Given the same manifest and container, every run produces the same tokens — verified across 3 models, 2 servers, and 8.88 million tokens.

## Results

**157/157 cross-server comparisons match (100%)** across two independent NVIDIA GH200 480GB instances on Lambda Cloud:

| Model | Type | Repeated | Diverse | Tokens |
|-------|------|----------|---------|--------|
| Qwen3-1.7B | Dense transformer | 20/20 match | 34/34 match | 1.6M |
| Qwen3-30B-A3B | Mixture of Experts | 20/20 match | 34/34 match | 2.0M |
| Mistral-7B-Instruct-v0.3 | Dense transformer | 20/20 match | 34/34 match | 2.0M |

Each chunk is 30,000 tokens of greedy decoding (temperature=0). Same container image on both servers, same seed, same config.

## Architecture

```
                              Deterministic Serving Stack
 ┌──────────────────────────────────────────────────────────────────────┐
 │                                                                      │
 │  ┌──────────┐    ┌──────────┐    ┌──────────────────────────────┐   │
 │  │ Manifest │───>│ Resolver │───>│ Resolved manifest + Lockfile │   │
 │  │ (author) │    │          │    │ (pinned revisions, digests)  │   │
 │  └──────────┘    └──────────┘    └───────────────┬──────────────┘   │
 │                                                  │                   │
 │                                                  v                   │
 │  ┌────────────────────────────────────────────────────────────────┐  │
 │  │                    Nix Container Image                         │  │
 │  │  ┌──────────────────────────────────────────────────────────┐  │  │
 │  │  │ Proxy Server (modules/inference/server/main.py)                        │  │  │
 │  │  │  POST /manifest ── validate schema                       │  │  │
 │  │  │                  ── verify GPU model, count, driver       │  │  │
 │  │  │                  ── verify model file digests             │  │  │
 │  │  │                  ── start vLLM with manifest settings     │  │  │
 │  │  │  GET  /manifest ── return active config + health          │  │  │
 │  │  │  POST /v1/...   ── proxy to vLLM + capture log           │  │  │
 │  │  └──────────────────────────┬───────────────────────────────┘  │  │
 │  │                             │                                  │  │
 │  │                             v                                  │  │
 │  │  ┌──────────────────────────────────────────────────────────┐  │  │
 │  │  │ vLLM 0.17.1 (VLLM_BATCH_INVARIANT=1, --enforce-eager)  │  │  │
 │  │  │  --model, --revision, --seed, --dtype,                   │  │  │
 │  │  │  --attention-backend, --max-model-len, ...               │  │  │
 │  │  │  (every manifest field passed as CLI flag or env var)     │  │  │
 │  │  └──────────────────────────────────────────────────────────┘  │  │
 │  └────────────────────────────────────────────────────────────────┘  │
 │                                                                      │
 │  ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────────┐  │
 │  │  Runner  │───>│ Capture  │───>│ Run Bundle│───>│   Verifier   │  │
 │  │(tokens,  │    │(request/ │    │(observ-   │    │(compare two  │  │
 │  │ logits,  │    │ response │    │ ables,    │    │ bundles via  │  │
 │  │ frames)  │    │ logging) │    │ frames,   │    │ comparison   │  │
 │  │          │    │          │    │ provenance│    │ config)      │  │
 │  └──────────┘    └──────────┘    └───────────┘    └──────────────┘  │
 └──────────────────────────────────────────────────────────────────────┘
```

## Quick Start (reviewers)

Bring up an NVIDIA H100 instance with the standard CUDA 12.8 AMI (Lambda Cloud's `gpu_1x_h100_sxm5` and `gpu_1x_h100_pcie` work as-is; GH200 also works), then:

```bash
git clone https://github.com/luke-marks0/deterministic_serving_stack
cd deterministic_serving_stack
./scripts/demo.sh
```

`scripts/demo.sh` builds a venv (cu128 torch + vLLM 0.17.1), resolves the audit-enabled smoke manifest at `experiments/e2e-audit/scripts/smoke.manifest.json` (declares H100 hardware, Qwen3-1.7B, 2 short prompts), starts the deterministic server, and runs the audit replay loop:

1. `POST /run` — server runs the manifest's requests and returns per-output-token HMAC commitments
2. `POST /replay` at random token positions — server re-runs each request truncated to the challenged position and recomputes the commitment
3. Negative test — a forged commitment must not match

Expected output ends with `ALL PASS`. Total wall time from `git clone` to `ALL PASS`: ~3 minutes (~90s pip install, <5s resolver/builder, ~30s vLLM model load, ~10s audit).

Requirements:
- NVIDIA GPU with compute capability ≥ 9.0 (H100, GH200, etc.) — batch invariance kernels need this
- ~5 GB free GPU memory (Qwen3-1.7B in bf16)
- Outbound internet for the Hugging Face download

### No GPU? Run the mock pipeline (wiring check)

Install the small CPU-only deps, then run the artifact spine on the mock backend —
a wiring check (no model download, no network), **not** a determinism proof:

```bash
uv sync   # installs the pinned CPU/test deps from uv.lock

tmp=$(mktemp -d)
.venv/bin/python3 modules/inference/resolver/main.py --manifest modules/inference/manifests/qwen3-1.7b.manifest.json \
  --lockfile-out $tmp/lock.json
.venv/bin/python3 modules/build/builder/main.py --lockfile $tmp/lock.json --lockfile-out $tmp/built.json
.venv/bin/python3 modules/inference/runner/main.py --manifest modules/inference/manifests/qwen3-1.7b.manifest.json \
  --lockfile $tmp/built.json --out-dir $tmp/run --mode mock
# Produces a run bundle with tokens, logits, and deterministic network frames.
# (Add --resolve-hf to the resolver to re-resolve revisions against live HF; needs network + huggingface_hub.)
```

Or compose the same spine in a few lines via a recipe — see
[`workflows/`](workflows/).

## How It Works

**Manifest** declares the full workload: model (pinned to HF commit SHA), runtime config (seed, dtype, attention backend, batch invariance), hardware requirements, requests, and comparison criteria.

**Resolver** pins everything to immutable references: resolves HF revisions, enumerates model files with per-file SHA256 digests, produces a lockfile.

**Nix container** pins the entire software stack: vLLM, PyTorch, CUDA toolkit, Triton, all Python deps. Same flake = same container = same behavior on any machine.

**Server** validates the manifest against the runtime (GPU model/count, driver version, CUDA version, model file digests), then starts vLLM with every manifest field passed as a CLI flag or env var.

**Runner** generates a run bundle containing tokens, logits, and deterministic L2 network frames (constructed by a simulated TCP/IP stack from the inference output).

**Verifier** compares two run bundles using the manifest's comparison config (exact match for tokens, tolerance for logits, SHA256 for network egress).

## What Makes It Deterministic

| Layer | How |
|-------|-----|
| **Software** | Hermetic Nix container — identical binary on every machine |
| **Model weights** | HF commit SHA pinned, per-file SHA256 verified before serving |
| **CUDA/cuBLAS** | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `VLLM_BATCH_INVARIANT=1` |
| **Attention** | `--enforce-eager` (no CUDA graphs), fixed attention backend |
| **Scheduling** | Greedy decoding (temperature=0), fixed seed |
| **Network frames** | Simulated TCP/IP stack with fixed MSS segmentation, software checksums, no offloads |

## Demos

- **Prover ↔ Verifier protocol** — wire-protocol demo that detects hidden training and exfiltration from external evidence alone. See [experiments/prover-verifier-demo/reports/memo.md](experiments/prover-verifier-demo/reports/memo.md) (CPU-only; `cd experiments/prover-verifier-demo && ./demo.sh --quick`).

## Capabilities

The stack is organized **by function**. Each capability has a documented
interface ([`modules/`](modules/)); [`workflows/`](workflows/) is the recipe book
that composes them.

> 💡 **New here, or looking for ideas of what to build?** See
> [**Ideas & use cases**](#ideas--use-cases) — a plain-language tour of what this
> can do.

| Capability | What it does | Start here |
|---|---|---|
| [build](modules/build/) | Hermetic, reproducible runtime + OCI image | `nix build .#oci` |
| [inference](modules/inference/) | Bitwise-deterministic vLLM (the c3 config) | `modules/inference/` |
| [network](modules/network/) | Deterministic L2 egress frames | `modules.network.egress_frames(...)` |
| [memory](modules/memory/) | PoSE memory wipe + erasure attestation | `modules/memory/` |
| [attestation](modules/attestation/) | Matmul / token / replay verification | `modules/attestation/verifier`, `modules/attestation/freivalds` |
| [utils](modules/utils/) | Provisioning, replay server, helpers | `scripts/deploy/`, `scripts/lambda/lambda_cli.py` |

Compose them in a few lines via [`modules.Pipeline`](modules/pipeline.py):

```python
from modules import Pipeline
report = (Pipeline.from_manifest("modules/inference/manifests/qwen3-1.7b.manifest.json")
          .resolve().build().run("/tmp/a").run("/tmp/b").verify())
assert report["status"] == "conformant"
```

See the [capability map](modules/README.md). (Design and implementation plans
live on the `experiments` branch.)

## Ideas & use cases

> **In one sentence:** this project makes AI systems *reproducible and provable* —
> two different computers running the same model give the **exact same answer,
> bit for bit** — which lets you prove what an AI system actually did.

Each idea below is written plainly first, with a pointer for engineers who want to
jump straight to the code.

### Why reproducibility matters (the plain version)

Normally, run an AI model twice and you can get slightly different answers — even
on the same input. That tiny wobble makes it impossible to *prove* anything: you
can't tell an honest mistake from tampering, or reproduce a result exactly.

This stack removes the wobble. Same model + same input → **identical output,
every time, on any matching machine.** Once outputs are exactly reproducible, you
can audit them, verify them, and share them.

### Ideas

#### 🔁 "Two servers, identical answers"
**For everyone:** run your AI service on two independent machines and get
byte-for-byte identical results — proof the service is behaving consistently and
hasn't been quietly changed.
**→ Engineers:** `workflows/deterministic_inference_server.py`

#### 🕵️ Catch a model doing something it shouldn't
**For everyone:** check whether an AI service that's *supposed* to just answer
questions is secretly training on your data or smuggling information out — using
only outside evidence, without trusting the operator.
**→ Engineers:** the prover-verifier demo + `workflows/verified_inference.py`

#### ✅ Prove the math was actually done
**For everyone:** get a cheap, independent receipt that the heavy computation a
provider charged you for was really performed correctly.
**→ Engineers:** matmul attestation (`modules/attestation`, Freivalds' algorithm)

#### 📒 Share an experiment as a single file
**For everyone:** instead of writing a page describing "here's what I ran,"
hand a colleague one short file they can run to reproduce your exact workload.
**→ Engineers:** the [recipe book](workflows/) (`workflows/`)

#### 🎯 Reproducible fine-tuning (LoRA)
**For everyone:** fine-tune a model in a way someone else can reproduce exactly,
down to the environment it ran in.
**→ Engineers:** `workflows/deterministic_lora_training.py`

#### 📡 Tamper-evident network traffic
**For everyone:** make the data a server sends over the wire perfectly
predictable, so any deviation is a red flag.
**→ Engineers:** `modules/network` (deterministic egress frames)

#### 🧹 Prove a machine wiped its memory
**For everyone:** get cryptographic proof that a computer actually erased
sensitive data from its memory, rather than just claiming it did.
**→ Engineers:** `modules/memory` (Proof of Secure Erasure)

#### 🏗️ Reproducible builds
**For everyone:** rebuild the exact same software environment from scratch and
get an identical result — the foundation everything else rests on.
**→ Engineers:** `modules/build` (`nix build .#oci`)

### Try it in 30 seconds (no GPU needed)

```bash
uv run python3 workflows/deterministic_inference_server.py --mode mock
# mode          : mock (no GPU) — wiring smoke test, NOT a determinism proof
# verify status : conformant
# egress frames : 1 (reproducible: True)
```

`--mode mock` runs the whole pipeline on a CPU stub — a wiring check, not a
determinism proof (the two mock runs match by construction). To actually *prove*
bitwise determinism, run `--mode vllm` on a GPU box (see [`scripts/demo.sh`](scripts/demo.sh)).

### Go deeper

- **[The recipe book](workflows/)** — runnable, copy-pasteable workflows
- **[Capability modules](modules/)** — the building blocks, each with a documented interface
- **[How it's organized](#repository-structure)** — the layout below

### Have an idea?

These are just starting points. If you have a workload you'd like to make
reproducible or verifiable, the building blocks in [`modules/`](modules/) are
meant to be combined — open an issue or a draft recipe and let's talk.

## Repository Structure

Organized **by function** — each capability physically owns its code:

```
modules/                Capability layer — each module owns its code, plus shared core/ + Pipeline
  build/                Hermetic runtime: builder/ + lockfiles/ + nix/   (flake.nix + flake.lock live at root)
  inference/            Deterministic vLLM — the c3 config
    server/             Proxy server with POST/GET /manifest endpoint
    resolver/           Manifest + HF resolution -> lockfile
    runner/             Manifest + lockfile -> run bundle (mock or vLLM)
    capture/            Server capture log -> run bundle
    manifest/           Pydantic manifest model (typed validation)
    manifests/          Model manifests (Qwen3, Mistral-Large2, DBRX, Llama4-Scout, ... + multinode)
  network/              networkdet/ (sim TCP/IP frame construction) + native/libnetdet/ (DPDK transmit)
  attestation/          freivalds/, e2e/, proverdet/ + verifier/ (+ verifier_cli/server) + prover/
  memory/               PoSE memory-wipe facade (over experiments/memory_wipe)
  utils/                Provisioning / replay helpers (re-exports core/common)
  core/                 Shared: common/ (canonical JSON, SHA256, schema validation, HF resolution)
                        + schemas/ (JSON Schema contracts: manifest, lockfile, run_bundle, verify_report, attestation/replay)
workflows/              Recipe book — runnable compositions of the modules
experiments/            Experiments product code/gates/demos depend on (research-only ones live on the `experiments` branch)
scripts/deploy/         Lambda / vast / warden provisioning (utils-owned)
tests/conformance/      Spec conformance catalog + release blockers (read by CI)
flake.nix, flake.lock   Hermetic build entrypoint + pin (at root: src=self packages repo-wide code; callers invoke `.#`)
```

## Container image

Building from this checkout is the canonical, reproducible path: `nix build .#oci`
produces `deterministic-serving-runtime:<git-rev>`. CI also publishes a
digest-tagged image to GHCR on every push to `main` —
`ghcr.io/<owner>/deterministic-serving:<git-sha>` (see
[`.github/workflows/nix-build.yml`](.github/workflows/nix-build.yml)). To run from
a container, build and load into Docker:

```bash
nix build .#oci
docker load < result
docker run -d --name vllm-server --gpus all --privileged \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$PWD:/workspace" -p 8000:8000 \
  deterministic-serving-runtime:dev \
  --manifest /workspace/experiments/e2e-audit/scripts/smoke.manifest.json \
  --skip-boot-validation
```

The NVIDIA Container Toolkit must be installed and configured as Docker's default runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
```

Troubleshooting:

| Symptom | Fix |
|---------|-----|
| `Failed to infer device type` | Add `--privileged -e NVIDIA_DRIVER_CAPABILITIES=all` |
| `No CUDA GPUs are available` | Add `--privileged` |
| `Can't initialize NVML` | Set `"default-runtime": "nvidia"` in daemon.json |
| `GLIBC_2.38 not found` | Don't set `LD_LIBRARY_PATH` to host system paths |

## Building from Source

```bash
# Build the hermetic runtime closure
nix build .#closure

# Build the OCI image
nix build .#oci

# Load into Docker
docker load < result
```

## CI Gates

| Gate | What it runs | Command |
|------|-------------|---------|
| PR | lint + schema + unit/integration | `make ci-pr` |
| Main | + e2e + determinism + nix closure | `make ci-main` |
| Nightly | + chaos + long-run | `make ci-nightly` |
| Release | + release contracts | `make ci-release` |
