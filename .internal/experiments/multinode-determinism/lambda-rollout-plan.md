# D6 on Lambda: Staged Multi-Node Determinism Rollout

> Author's note to the engineer picking this up:
>
> You are about to spend real money on GPUs trying to prove a subtle property about a system you've never seen. This document assumes you know nothing about our codebase or the determinism problem, but that you can write Python and run a shell. Read it top to bottom **before** you provision a single instance. Each task is small, has a verification step, and ends with a commit. Do not skip the verification steps. Do not "improve" the plan on the first pass — if you see something that bothers you, write it in the experiment log and keep going.

---

## TL;DR

Three phases, each gated by the previous:

1. **Phase 1** — One H100, one container, one tiny inference. Prove the container works on Lambda's hardware.
2. **Phase 2** — Two H100s, a Ray cluster, a real distributed (PP=2) inference run. Then prove with multiple independent checks that distribution is *actually happening*, not faked by replicated work. (PP=2, not TP=2 — so the runner's NCCL pinning code path is the same one Phase 3 will exercise.)
3. **Phase 3** — Four H100s, the full D6 experiment from `docs/plans/d6-multinode-distributed-determinism.md` against Mistral Large 2 (dense) and DBRX (MoE).

You will keep an experiment log at `experiments/d6-lambda-rollout-log.md` that records every config, every setback, and every milestone. The log is git-tracked and committed alongside code changes.

Total expected wall time: **6–12 hours** including capacity polling and two large per-node downloads (Mistral Large 2 ~240 GB + DBRX ~265 GB ≈ 505 GB per node). Total expected GPU spend: **$60–150**. You can blow that budget very quickly if you get sloppy with teardown — be paranoid about leaked instances.

---

## ⚠️ If You Stop For Any Reason, Terminate Everything

If you walk away from your laptop, log off for the night, hit a blocker, or lose context for *any* reason — run this first:

```bash
python3 scripts/lambda/lambda_cli.py terminate-all
# verify in the Lambda web UI: https://cloud.lambda.ai/instances
```

Lambda H100s are $4–17/hr depending on size. A forgotten 4-node cluster overnight is real money. The phase boundaries in this plan are deliberately placed where you can stop, terminate, and pick up later from a fresh poll. **There is no state on the instances that matters between phases** — the experiment log on your laptop is the source of truth. Re-acquiring nodes is annoying but cheap; leaking them is not.

---

## Why a Staged Rollout (and not the all-at-once plan in `d6-multinode-distributed-determinism.md`)

The original D6 plan assumes you can rent 4× H100s, set them up, and run experiments in one straight shot. We tried that and it fell apart for two reasons:

1. **Lambda H100 capacity is intermittent.** You poll, you grab what you can get, you don't get all 4 at the same time. By the time the 4th lands, the 1st might be 30 minutes into the meter. If you're going to burn money waiting, you want to be sure the burn rate is buying you something.
2. **Failure modes compound.** If you launch 4 nodes and then discover the container is broken, you've burned 4× the cost discovering one bug. Worse, you don't know whether the bug is in the container, the cluster, the model, the manifest, or your own setup. Staged rollout isolates one variable per phase.

The staging is also a hard rule about what you're allowed to skip:

- **You may not start Phase 2 until Phase 1's determinism check passes.** If the same prompt produces different tokens on a single GPU, multi-node will not save you.
- **You may not start Phase 3 until Phase 2's anti-cheat checks pass.** "It looked like it worked" is not a passing grade. We are specifically trying to rule out the failure mode where each GPU runs the whole model and the answer happens to match.
- **You may not skip the experiment log.** Not even "I'll fill it in later." Fill it in as you go. Future-you will thank present-you when something inexplicably stops working at 3am.

---

## Mental Model: What Are We Actually Proving?

Read this section even if you think you know it. The whole experiment hinges on understanding *what counts as a pass.*

### What "deterministic inference" means here

When we say a model is deterministic, we mean: given identical inputs, identical configuration, and identical hardware, the model produces **bitwise identical** output tokens. Not "approximately the same." Not "same to four decimal places." The same int64 token IDs in the same order.

This is hard to achieve because LLM inference touches dozens of nondeterminism sources:

- CUDA kernel autotuning picks different algorithms based on timing.
- Floating-point reductions are not associative; `(a+b)+c ≠ a+(b+c)` for floats. Reduction order matters.
- Atomic adds on GPUs commit in undefined order under contention.
- Batched requests share kernel launches; padding and ordering shift the math.
- NCCL collectives across nodes can pick different algorithms (Ring vs Tree) and protocols (Simple vs LL).
- Network packet arrival order can shift reduction order.

For each one, there is a knob to pin it. The deterministic serving stack's job is to flip every knob and **prove** the system bites down. That proof is reproducible inference: run twice, get the same tokens; or run the same prompts in shuffled order with a different batch size, get the same tokens per request.

### What D6 specifically adds

D1–D5 already proved single-GPU and single-node multi-GPU determinism. D6 is the hardest case: **GPUs on different physical machines, talking over TCP**. Why hardest:

- Single-node multi-GPU uses NVLink: hardware-ordered, deterministic packet delivery, ~600 GB/s.
- Multi-node uses NCCL over TCP sockets: kernel network stack, switches, NICs, packet reordering.

If we can pin NCCL to `Ring` algorithm + `Simple` protocol over TCP and get bitwise reproducibility, we've covered every realistic production topology. This is the deliverable of D6.

The two parallelism strategies we test:

- **Pipeline Parallel (PP=4)** — Each node holds 1/4 of the model's layers. Cross-node traffic is a single point-to-point send/recv between adjacent stages per token. Lower bandwidth requirement.
- **Tensor Parallel (TP=4 over TCP)** — Each node holds 1/4 of every layer. Every layer does an all-reduce across all 4 nodes. Massive bandwidth requirement, much slower over TCP than NVLink, but a strict superset of the single-node TP proof from D4.

Both must be deterministic for D6 to pass.

### What the engineer's job is, concretely

You will:
1. Provision Lambda H100 instances.
2. Run a known-good container that has vLLM + Ray + the runner code baked in.
3. Form a Ray cluster across nodes (this is the new bit; we did not do this in D1–D5).
4. Run the deterministic runner with multinode manifests.
5. Compare outputs of two runs and assert they're bitwise identical.
6. Repeat under varied configurations (same-config repeat, batch+order shuffle, TP vs PP).
7. Write a report.

You are not writing new vLLM code. You are not writing new runner code (most of it is already there). Your code contributions in the happy path are limited to:
- A Lambda helper script (Python, ~150 LoC).
- A handful of small verification scripts (anti-cheat checks).
- One or two new pytest tests (if needed).
- Documentation and the experiment log.

If you find yourself rewriting `cmd/runner/vllm_runner.py`, **stop** and check the experiment log to see if you've drifted off the plan.

---

## Codebase Tour (10 minutes)

The repo is bigger than it looks. Here's the minimum you need to know to do D6. You should at least open each of these files and skim them once before Phase 1.

### Concepts

- **Manifest** (`manifests/*.manifest.json`) — Declarative description of an experiment: which model, which hardware, which seeds, which serving config, which prompts. Not enough to reproduce by itself; needs a lockfile.
- **Lockfile** (`lockfiles/*.lockfile.json`, may need to be generated) — Resolved digests of every artifact: weights shards by sha256, config files, tokenizer revision. Together with the manifest, defines an exactly-reproducible run.
- **Runner** (`cmd/runner/main.py`) — Reads a manifest + lockfile, sets the deterministic environment, calls the backend (vLLM, network capture, etc.), and writes structured **observables** (`observables.json`).
- **Observables** — What we compare across runs. Includes `request_outputs[].tokens` (the int64 token IDs), `engine_events`, `frames`, environment snapshot. Token-level comparison is the gold-standard pass criterion.
- **Replica** — A single execution unit. The runner writes `replica-<id>/observables.json`. Multi-replica runs are how D3 proves cross-node determinism with the same config.

### Files you must read

| File | What it does | When you need it |
|------|--------------|------------------|
| `docs/plans/d6-multinode-distributed-determinism.md` | The original D6 plan that assumed 4 nodes available all at once. Your test scenarios come from here. | Phase 0 |
| `experiments/MULTI_GPU_DETERMINISM_REPORT.md` | The D4 report. Read the "Conclusion" section to understand what passing looks like. | Phase 0 |
| `cmd/runner/main.py` | Runner CLI entry point. Validates manifest + lockfile, dispatches to backend, writes observables. ~650 lines. Read top to `_env_or_default` to understand the CLI. | Phase 1 |
| `cmd/runner/vllm_runner.py` | The vLLM backend. **Crucially**, lines 15–36 set the NCCL env vars when `pp_size>1` or `VLLM_MULTI_NODE` is set. Read this carefully — it's the heart of D6. | Phase 1 |
| `cmd/runner/dispatcher.py` | Picks which backend to call based on `--mode`. Trivial. | Optional |
| `pkg/manifest/model.py` | Pydantic models for the v1 manifest. `ServingEngine` includes `tensor_parallel_size`, `pipeline_parallel_size`, and `distributed_executor_backend`. | Phase 1 |
| `schemas/manifest.v1.schema.json` | JSON schema for manifest validation. The runner validates against this on every run. | If you edit a manifest |
| `manifests/dbrx-pp4-multinode.manifest.json` | One of the four manifests we'll run in Phase 3. Already created. | Phase 3 |
| `manifests/mistral-large2-pp4-multinode.manifest.json` | Same. | Phase 3 |
| `scripts/ci/d6_multinode_determinism.sh` | The existing harness that runs the 3 D6 tests. You will execute this in Phase 3. Read it now to understand what it does. ~210 lines. | Phase 3 |
| `flake.nix` | The Nix recipe that built the container. Read the `ociImage` definition (around line 352) so you understand what's in the container and what isn't. | Reference |

### Files you should know exist but probably won't touch

| File | What it does |
|------|--------------|
| `cmd/server/main.py` | vLLM API server CLI. Only relevant if you're testing the online serving path. We use the offline runner for D6. |
| `cmd/builder/main.py` | The lockfile builder. You may need to invoke this if no lockfile exists for our models. |
| `cmd/resolver/` | Resolves HuggingFace artifacts to digests. Used by builder. |
| `cmd/verifier/` | Compares observables across runs. We use a smaller inline comparison in `d6_multinode_determinism.sh`. |
| `cmd/coordinator/` | Multi-replica orchestration. Not needed for D6. |
| `tests/determinism/` | Pytest suite for D0–D4 determinism. Read `test_d4_batch_order_invariance.py` for the comparison pattern. |
| `tests/integration/test_runner_hardware_conformance.py` | Hardware conformance check. The runner errors out if the hardware doesn't match the manifest profile. **You will hit this** if `strict_hardware: true`. Our multinode manifests set it to `false`. |

### The Container

`ghcr.io/derpyplops/deterministic-serving:multinode` — public, pulls without auth, ~6.6 GB.

It contains:
- Python 3.12 (Nix-built)
- PyTorch 2.10.0 with CUDA support
- vLLM 0.17.1 from source (pinned)
- Ray 2.54.0
- FlashAttention v3
- The deterministic serving stack code at `/workspace` (Nix store path actually, see `flake.nix`)

It does **not** contain:
- An SSH server (this is why we can't use it as a vast.ai base image — vast.ai's proxy needs sshd inside the container)
- A package manager (it's Nix; `pip install` won't work)
- HuggingFace cached models (you mount these from the host)

The default `ENTRYPOINT` is `python3 cmd/server/main.py`. For our purposes you almost always want `--entrypoint /bin/bash` so you can drop into a shell and invoke `python3 cmd/runner/main.py` yourself.

**Critical run flags** (memorize these, they appear in every Phase):
```bash
docker run \
  --rm \
  --network host \
  --gpus all \
  --shm-size 10.24g \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTHONPATH=/workspace \
  --entrypoint /bin/bash \
  ghcr.io/derpyplops/deterministic-serving:multinode \
  -c "<your command here>"
```

`--network host` is non-negotiable: Ray and NCCL need to bind to the host's interfaces, not a Docker bridge. This is *the* lesson of vast.ai's failure — Docker NAT breaks Ray/NCCL across hosts. Lambda gives you a real VM, so `--network host` works the way you'd hope.

---

## Prerequisites

Run through this checklist before Phase 0. If any item fails, fix it before proceeding.

### Local environment

```bash
# 1. You're on the multi-gpu-determinism branch
git -C ~/projects/deterministic_serving_stack rev-parse --abbrev-ref HEAD
# Expected: multi-gpu-determinism

# 2. uv is installed (we use uv, not pip/pipx/apt)
uv --version

# 3. vastai CLI is installed (used by an earlier session; harmless leftover)
vastai --version || echo "ok if not present, we're not using vast"

# 4. gh is logged in
gh auth status

# 5. You have an SSH private key locally
ls ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub

# 6. LAMBDALABS_API_KEY is exported
echo "${LAMBDALABS_API_KEY:?LAMBDALABS_API_KEY must be set}" | head -c 12 ; echo
```

If `LAMBDALABS_API_KEY` is missing, get it from the Lambda Cloud console (`https://cloud.lambda.ai/api-keys`) and put it in `~/.zshrc`:
```bash
export LAMBDALABS_API_KEY="secret_..."
```
Then `source ~/.zshrc`.

### Repo state

```bash
# Working tree should be clean enough to commit
cd ~/projects/deterministic_serving_stack
git status

# These four manifests must exist (created in a prior session)
ls manifests/{dbrx,mistral-large2}-{pp4,tp4}-multinode.manifest.json
```

If the manifests are missing, regenerate them following the recipe in the existing `mistral-large2-tp4.manifest.json` (set `pipeline_parallel_size`/`tensor_parallel_size` and add `"distributed_executor_backend": "ray"`). The DBRX multinode manifests are stubs — their HF model shard list is populated by `cmd/resolver/main.py --resolve-hf` in Task 3.5.

### Required reading

Before Task 0.1, read these in this order. Set a 30-minute timer.

1. `docs/plans/d6-multinode-distributed-determinism.md` — the original D6 plan.
2. `experiments/MULTI_GPU_DETERMINISM_REPORT.md` — D4's results, especially the conclusion and the failure modes section.
3. `cmd/runner/vllm_runner.py` lines 1–80 — the NCCL env var pinning logic.
4. `scripts/ci/d6_multinode_determinism.sh` — the test harness you'll eventually run.

---

## How to Use This Plan

- **Work the tasks in order.** Each one is sized to be 5–30 minutes of focused effort. If a task is taking >1 hour, stop, log what's stuck, ask for help.
- **Update the experiment log after every task**, not at the end of the phase. The log is your insurance against having to repeat work.
- **Commit after every task.** Conventional commit messages with a `d6: ` prefix. Example: `d6: phase 1 task 1.5 — first inference on Lambda H100`.
- **The verification step is part of the task.** A task is not done until its verification passes.
- **Do not optimize.** No premature abstractions, no fancy retries, no self-healing. If the script fails, fix what's broken and run it again. The plan is the abstraction.
- **If you must deviate**, write the deviation in the experiment log first. Include the *why*. Then deviate.

### Anti-patterns to avoid

- Adding error handling to scripts that won't exist tomorrow. Catch errors at the boundary (Lambda API, SSH connection); let everything else crash loudly.
- Building a "framework" for the verification scripts. Write four scripts, not one with five plug-in checks.
- Using `time.sleep(N)` instead of "if you're polling, log the timestamp every iteration so we can see how long it took."
- Reformatting code that's already in the repo.
- Renaming things to match your taste. The names exist for reasons that may not be obvious. Ask first.

---

## The Experiment Log

### File location

Create `experiments/d6-lambda-rollout-log.md`. This is a single-file append-only journal. Git-track it and commit alongside the work.

### Format

Use `##` headers for major events (task start, milestone, setback, end of phase). Use `###` for sub-events. Always include UTC timestamps. Include configurations *verbatim* — copy-paste, don't summarize.

### Template (start of file)

```markdown
# D6 Lambda Rollout — Experiment Log

This is an append-only journal for the staged D6 rollout on Lambda Cloud.
Every major action, configuration, setback, and milestone is logged here.

Engineer: <your name or handle>
Started: <date>
Plan: docs/plans/d6-lambda-staged-rollout.md

---

## Phase 0: Bootstrap

### <ISO timestamp> — Started Phase 0

(insert sub-events as they happen)

```

### Event types and naming convention

Use these prefixes consistently so the log can be grepped:

- `MILESTONE:` — A passing verification. Always include what was tested and what passed.
- `SETBACK:` — Anything that didn't go to plan. Include the error message verbatim, what you tried, and what fixed it (if anything).
- `CONFIG:` — A configuration change that affects future runs. Manifest edits, NCCL env vars, etc.
- `DECISION:` — A choice that wasn't pre-decided in the plan. Include the alternatives you considered.
- `COST:` — A cost-relevant event: instance launched, instance terminated, capacity status.

### Example entry

```markdown
### 2026-04-13T14:42Z — MILESTONE: first inference on Lambda H100

Successfully ran Qwen3-0.6B inference inside the container on instance
`abc123` (gpu_1x_h100_sxm5, us-west-1).

Container ID: 8f3d58eceb0a
GPU: NVIDIA H100 80GB HBM3 (driver 580.126.20, CUDA 13.0)
Wall time: 7.8s (engine init) + 0.99s (1 prompt, 20 tokens)
Output tokens (decoded): " a philosophical question that has long been..."
First 5 token IDs: [1102, 264, 41733, 3405, 429]

Verification: re-ran the same script, token IDs matched exactly.
```

### When you get stuck

When a setback happens, write the entry **before** trying to fix it. The act of writing it down often reveals the bug. Even if it doesn't, you save your future self from "wait, did we already try X?"

---

## Phase 0: Bootstrap

Goal: get the local environment, Lambda API access, and supporting scripts in place. No GPU instances are launched in Phase 0. Total effort: ~45 minutes.

### Task 0.1: Read the existing plans and reports

**Goal:** Know what passing looks like before you try to make anything pass.

**Steps:**
1. Open `docs/plans/d6-multinode-distributed-determinism.md`. Read the "Motivation" and "Execution Plan" sections.
2. Open `experiments/MULTI_GPU_DETERMINISM_REPORT.md`. Read the conclusion. Note the format of the results table.
3. Open `cmd/runner/vllm_runner.py`. Read the `_set_deterministic_env` function (top of file). Notice that it conditionally sets `NCCL_NET=Socket` and friends when `pp_size > 1` or `VLLM_MULTI_NODE` is set — that's the multinode pinning.

**Verification:**
You can answer these questions without re-reading:
- What does `pp_size > 1` cause vLLM_runner to set in the env?
- What's the difference between PP=4 and TP=4 over TCP, in terms of cross-node bandwidth?
- What does the D4 report say is the pass criterion (token-level match? logits match? hash match?)

**Commit:** none — reading only.

---

### Task 0.2: Create the experiment log

**Goal:** Have a place to write things down before you have anything to write down.

**Steps:**
```bash
cd ~/projects/deterministic_serving_stack
mkdir -p experiments
cat > experiments/d6-lambda-rollout-log.md <<'EOF'
# D6 Lambda Rollout — Experiment Log

This is an append-only journal for the staged D6 rollout on Lambda Cloud.
Every major action, configuration, setback, and milestone is logged here.

Plan: docs/plans/d6-lambda-staged-rollout.md

---

## Phase 0: Bootstrap

EOF
```

Add a first entry:

```markdown
### <ISO timestamp> — Started Phase 0

Working through docs/plans/d6-lambda-staged-rollout.md.
Local environment verified: uv, gh auth, LAMBDALABS_API_KEY set.
```

**Verification:**
```bash
test -f experiments/d6-lambda-rollout-log.md && echo OK
```

**Commit:**
```
git add experiments/d6-lambda-rollout-log.md
git commit -m "d6: phase 0 task 0.2 — start experiment log"
```

---

### Task 0.3: Verify Lambda API access

**Goal:** Confirm `LAMBDALABS_API_KEY` works and you can read account state.

**Steps:**
```bash
# Should return JSON with a "data" key listing instance types
curl -sf -u "$LAMBDALABS_API_KEY:" \
  https://cloud.lambdalabs.com/api/v1/instance-types \
  | python3 -m json.tool | head -20

# Should return JSON with a "data" key listing your instances (probably empty)
curl -sf -u "$LAMBDALABS_API_KEY:" \
  https://cloud.lambdalabs.com/api/v1/instances \
  | python3 -m json.tool

# Should return your registered SSH keys
curl -sf -u "$LAMBDALABS_API_KEY:" \
  https://cloud.lambdalabs.com/api/v1/ssh-keys \
  | python3 -m json.tool
```

**Verification:**
- All three calls return HTTP 200 (i.e., curl doesn't error out due to `-f`).
- The instance-types response contains `gpu_1x_h100_sxm5`.
- Note any existing SSH keys; you'll need their names in Task 0.4.

**Log entry:**
```markdown
### <timestamp> — Verified Lambda API access

Listed instance types (16 total), instances (0 running), ssh keys (N keys).
Existing key names: <names>
```

**Commit:** none — read-only.

---

### Task 0.4: Register your SSH key with Lambda

**Goal:** Lambda needs to know which public key to install on launched instances. Your CLAUDE.md mentions a key called `macbook 2025` — that's the one registered from a different machine. You need a key from *this* machine.

**Steps:**

```bash
# Already have one?
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "d6-rollout"

# Register it under a new name
PUBKEY=$(cat ~/.ssh/id_ed25519.pub)
curl -sf -u "$LAMBDALABS_API_KEY:" \
  -H "Content-Type: application/json" \
  -X POST https://cloud.lambdalabs.com/api/v1/ssh-keys \
  -d "{\"name\": \"d6-rollout\", \"public_key\": \"$PUBKEY\"}" \
  | python3 -m json.tool
```

If the response says the key already exists under a different name, that's fine — note the name and use it instead of `d6-rollout` going forward.

**Verification:**
```bash
curl -sf -u "$LAMBDALABS_API_KEY:" \
  https://cloud.lambdalabs.com/api/v1/ssh-keys \
  | python3 -c "import json,sys; print([k['name'] for k in json.load(sys.stdin)['data']])"
```

You should see `d6-rollout` (or the name you used) in the list.

**Log entry:**
```markdown
### <timestamp> — Registered SSH key with Lambda

Key name: d6-rollout
Public key fingerprint: <ssh-keygen -lf ~/.ssh/id_ed25519.pub>
```

**Commit:** none.

---

### Task 0.5: Write the Lambda helper script

**Goal:** One Python script that wraps every Lambda API call we'll make. We will call it from every phase. It is the *only* place where the API URL appears. **DRY.**

**Why a helper, not bash one-liners:** poll loops and JSON parsing in bash get ugly fast. Python is cleaner and more debuggable. We're not building a framework — this is a 150-line script with one job.

**Steps:**

Create `scripts/lambda/lambda_cli.py`:

```bash
mkdir -p scripts/lambda
```

Write the file with the contents below. Resist the urge to "improve" it. It does what it needs to do.

```python
#!/usr/bin/env python3
"""Lambda Cloud helper for the D6 rollout.

One place for every Lambda API interaction we need. DRY.

Subcommands:
    list                 List running instances (id, type, ip, status).
    keys                 List registered SSH keys.
    add-key NAME PUBKEY  Register an SSH public key.
    types                List instance types and current capacity.
    poll TYPE [opts]     Poll for capacity, launch when available.
    terminate ID         Terminate an instance.
    terminate-all        Terminate every running instance (with confirmation).

Auth: reads LAMBDALABS_API_KEY from env. No fallback; we don't want surprises.

Run with:  python3 scripts/lambda/lambda_cli.py <subcommand> [args...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from base64 import b64encode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_BASE = "https://cloud.lambdalabs.com/api/v1"


def _auth_header() -> dict[str, str]:
    key = os.environ.get("LAMBDALABS_API_KEY")
    if not key:
        sys.exit("LAMBDALABS_API_KEY is not set")
    creds = b64encode(f"{key}:".encode()).decode()
    return {
        "Authorization": f"Basic {creds}",
        "Content-Type": "application/json",
    }


def api(method: str, path: str, body: dict | None = None) -> dict:
    """Single chokepoint for every Lambda API call."""
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = Request(url, data=data, method=method, headers=_auth_header())
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        sys.exit(f"{method} {path} -> HTTP {e.code}: {e.read().decode()[:500]}")
    except URLError as e:
        sys.exit(f"{method} {path} -> network error: {e}")


# ───────────────────────── subcommands ─────────────────────────

def cmd_list(_args: argparse.Namespace) -> None:
    data = api("GET", "/instances")["data"]
    if not data:
        print("(no instances)")
        return
    for d in data:
        print(
            f"{d['id']:24s} "
            f"{d['instance_type']['name']:24s} "
            f"{d.get('ip', '-'):16s} "
            f"{d.get('status', '?')}"
        )


def cmd_keys(_args: argparse.Namespace) -> None:
    for k in api("GET", "/ssh-keys")["data"]:
        print(f"{k['id']}\t{k['name']}")


def cmd_add_key(args: argparse.Namespace) -> None:
    pubkey = open(args.pubkey_file).read().strip()
    out = api("POST", "/ssh-keys", {"name": args.name, "public_key": pubkey})
    print(json.dumps(out, indent=2))


def cmd_types(_args: argparse.Namespace) -> None:
    data = api("GET", "/instance-types")["data"]
    for name in sorted(data.keys()):
        info = data[name]
        regions = [r["name"] for r in info["regions_with_capacity_available"]]
        price = info["instance_type"]["price_cents_per_hour"] / 100
        gpus = info["instance_type"]["specs"]["gpus"]
        avail = ",".join(regions) if regions else "-"
        print(f"{name:30s} ${price:6.2f}/hr  gpus={gpus}  available_in: {avail}")


def cmd_terminate(args: argparse.Namespace) -> None:
    out = api(
        "POST",
        "/instance-operations/terminate",
        {"instance_ids": [args.instance_id]},
    )
    print(json.dumps(out, indent=2))


def cmd_terminate_all(_args: argparse.Namespace) -> None:
    data = api("GET", "/instances")["data"]
    if not data:
        print("(nothing to terminate)")
        return
    ids = [d["id"] for d in data]
    print("about to terminate:")
    for d in data:
        print(f"  {d['id']}  {d['instance_type']['name']}  {d.get('ip','-')}")
    if input("type 'yes' to confirm: ").strip() != "yes":
        sys.exit("aborted")
    out = api(
        "POST",
        "/instance-operations/terminate",
        {"instance_ids": ids},
    )
    print(json.dumps(out, indent=2))


def cmd_poll(args: argparse.Namespace) -> None:
    """Poll for capacity, launch one at a time."""
    target = args.count
    interval = args.interval
    region_filter = args.region
    instance_type = args.type
    ssh_key = args.ssh_key
    name_prefix = args.name_prefix

    print(
        f"polling for {target}× {instance_type} "
        f"(interval={interval}s, region={region_filter or 'any'})"
    )
    launched: list[str] = []
    iteration = 0
    while len(launched) < target:
        iteration += 1
        ts = time.strftime("%H:%M:%S")
        types = api("GET", "/instance-types")["data"]
        if instance_type not in types:
            sys.exit(f"unknown instance type: {instance_type}")
        regions = [
            r["name"]
            for r in types[instance_type]["regions_with_capacity_available"]
        ]
        if region_filter:
            regions = [r for r in regions if r == region_filter]
        if not regions:
            print(f"  [{ts}] iter {iteration}: no capacity ({len(launched)}/{target} launched)")
            time.sleep(interval)
            continue

        region = regions[0]
        body = {
            "region_name": region,
            "instance_type_name": instance_type,
            "ssh_key_names": [ssh_key],
            "name": f"{name_prefix}-{len(launched) + 1}",
        }
        try:
            out = api("POST", "/instance-operations/launch", body)
        except SystemExit as e:
            # Capacity can vanish between the check and the launch. Try again.
            print(f"  [{ts}] launch failed in {region}: {e}")
            time.sleep(interval)
            continue
        ids = out["data"]["instance_ids"]
        launched.extend(ids)
        print(f"  [{ts}] launched {ids} in {region}  ({len(launched)}/{target})")

    print(f"all {target} launched: {launched}")


# ───────────────────────── argument parser ─────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)
    sub.add_parser("keys").set_defaults(func=cmd_keys)
    sub.add_parser("types").set_defaults(func=cmd_types)

    p_addkey = sub.add_parser("add-key")
    p_addkey.add_argument("name")
    p_addkey.add_argument("pubkey_file")
    p_addkey.set_defaults(func=cmd_add_key)

    p_term = sub.add_parser("terminate")
    p_term.add_argument("instance_id")
    p_term.set_defaults(func=cmd_terminate)

    sub.add_parser("terminate-all").set_defaults(func=cmd_terminate_all)

    p_poll = sub.add_parser("poll")
    p_poll.add_argument("type", help="e.g. gpu_1x_h100_sxm5")
    p_poll.add_argument("--count", type=int, default=1)
    p_poll.add_argument("--region", help="restrict to one region")
    p_poll.add_argument("--interval", type=int, default=30)
    p_poll.add_argument("--ssh-key", default="d6-rollout")
    p_poll.add_argument("--name-prefix", default="d6-node")
    p_poll.set_defaults(func=cmd_poll)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

**Verification:**
```bash
python3 scripts/lambda/lambda_cli.py types | head -5
python3 scripts/lambda/lambda_cli.py list
python3 scripts/lambda/lambda_cli.py keys
```

All three should run cleanly. None should print a Python traceback.

**Commit:**
```
git add scripts/lambda/lambda_cli.py
git commit -m "d6: phase 0 task 0.5 — add Lambda CLI helper"
```

---

### Task 0.6: Smoke-test the helper without launching anything

**Goal:** Confirm the poll loop exits cleanly when you Ctrl-C it, and that the launch path *would* work if there were capacity. Never spend money in Phase 0.

**Steps:**

```bash
# Run the poll command for a non-existent capacity, then Ctrl-C after one iteration
python3 scripts/lambda/lambda_cli.py poll gpu_8x_h100_sxm5 --count 1 --interval 5
# Watch one iteration print the timestamp, then ^C
```

**Verification:**
- The poll loop prints a timestamped "no capacity" line.
- ^C exits cleanly (no traceback longer than 5 lines).

**Commit:** none.

---

### Task 0.7: End of Phase 0

Update the experiment log:

```markdown
### <timestamp> — END Phase 0: bootstrap complete

Lambda API reachable, SSH key registered, helper script in place and tested.
Proceeding to Phase 1.
```

**Commit:**
```
git add experiments/d6-lambda-rollout-log.md
git commit -m "d6: end phase 0 — bootstrap complete"
```

---

## Phase 1: Single-Node Smoke Test

Goal: rent ONE H100, pull the container, run ONE inference, prove it's deterministic on a single GPU. This phase exists so that when Phase 2 fails (and it will), you know it's not the container or the model. Total effort: ~45 minutes including capacity poll.

### Task 1.1: Poll for one H100

**Goal:** Get one H100 up. We try `gpu_1x_h100_sxm5` first; fall back to `gpu_1x_h100_pcie` if SXM is unavailable for >10 minutes.

**Pin the region if you can.** Phase 3 has a much faster download path (rsync) if all 4 nodes end up in the same Lambda region. Pick a region that's likely to have multiple H100s available — `us-east-1`, `us-east-3`, `us-west-1`, and `us-west-2` are common choices. You can drop the `--region` filter if capacity is scarce.

**Steps:**
```bash
# Try with a region pin first
python3 scripts/lambda/lambda_cli.py poll gpu_1x_h100_sxm5 --count 1 --interval 30 --region us-east-3

# If nothing is available there for several minutes, drop the pin
python3 scripts/lambda/lambda_cli.py poll gpu_1x_h100_sxm5 --count 1 --interval 30
```

If it sits at "no capacity" for 10+ iterations, ^C and try PCIe:
```bash
python3 scripts/lambda/lambda_cli.py poll gpu_1x_h100_pcie --count 1 --interval 30
```

When you see `launched [...] in <region>`, note the instance ID and ^C.

**Verification:**
```bash
python3 scripts/lambda/lambda_cli.py list
# Should show 1 instance, status: booting (or active)
```

**Log entry:**
```markdown
### <timestamp> — COST: launched 1× H100 on Lambda

Instance ID: <id>
Type: gpu_1x_h100_sxm5
Region: <region>
Cost: $4.29/hr (or $3.29 for PCIe)
Polling time: <minutes>
```

**Commit:** none.

---

### Task 1.2: Wait for the instance to become reachable

**Goal:** Lambda instances go from `booting` → `active`. SSH is only reliable once status is `active` AND port 22 is actually accepting connections.

**Steps:**
```bash
# Get the IP once status is active
INSTANCE_ID=<from previous task>
while true; do
  STATUS=$(python3 scripts/lambda/lambda_cli.py list | grep "$INSTANCE_ID" | awk '{print $4}')
  IP=$(python3 scripts/lambda/lambda_cli.py list | grep "$INSTANCE_ID" | awk '{print $3}')
  echo "$(date +%H:%M:%S) status=$STATUS ip=$IP"
  if [ "$STATUS" = "active" ] && ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$IP "echo ok" 2>/dev/null; then
    echo "READY: $IP"
    break
  fi
  sleep 15
done
```

(If you prefer, write this as a one-shot `wait_active` subcommand on the helper. Resist the urge unless you'll use it more than twice. YAGNI.)

**Verification:**
```bash
ssh ubuntu@<IP> "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"
# Expected output: NVIDIA H100 80GB HBM3, 5XX.YY.ZZ
```

**Log entry:**
```markdown
### <timestamp> — MILESTONE: SSH ready on Node 1

IP: <ip>
GPU: NVIDIA H100 80GB HBM3
Driver: <version>
Time from launch to SSH-ready: <minutes>
```

**Commit:** none.

---

### Task 1.3: Pull the container

**Goal:** Get `ghcr.io/derpyplops/deterministic-serving:multinode` onto the node.

**Why ahead of inference:** the pull is ~6.6 GB. You want to surface any registry/network errors *now*, not 20 minutes into a determinism test.

**Steps:**
```bash
ssh ubuntu@<IP> "docker pull ghcr.io/derpyplops/deterministic-serving:multinode"
```

If `docker` is missing on the Lambda image (it shouldn't be — Lambda's Ubuntu DL image has docker pre-installed), install it:
```bash
ssh ubuntu@<IP> "curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker ubuntu"
# log out, log back in to pick up the group, then retry the pull
```

**Verification:**
```bash
ssh ubuntu@<IP> "docker images ghcr.io/derpyplops/deterministic-serving:multinode --format '{{.Repository}}:{{.Tag}} {{.Size}}'"
# Expected: ghcr.io/derpyplops/deterministic-serving:multinode 6.6GB (or close)
```

**Log entry:**
```markdown
### <timestamp> — MILESTONE: container pulled on Node 1

Image: ghcr.io/derpyplops/deterministic-serving:multinode
Size: <size>
Pull time: <minutes>
```

**Commit:** none.

---

### Task 1.4: Drop into the container and verify GPU access

**Goal:** Confirm the container can see the GPU through `--gpus all`.

**Steps:**
```bash
ssh ubuntu@<IP>
# now on the Lambda VM
docker run --rm --gpus all --network host \
  --entrypoint /bin/bash \
  ghcr.io/derpyplops/deterministic-serving:multinode \
  -c 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"'
```

**Verification:**
- `nvidia-smi` reports the H100.
- Python prints `True 1 NVIDIA H100 80GB HBM3` (or similar).

If torch.cuda.is_available() is False: the container is not seeing the GPU. Diagnose:
- `nvidia-smi` on the host (outside the container) — does the host see it? It must.
- `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` — does a known-good image see it?

**Log entry:**
```markdown
### <timestamp> — MILESTONE: container has GPU access

Container can see 1× NVIDIA H100 80GB HBM3, torch.cuda.is_available()=True.
```

**Commit:** none.

---

### Task 1.5: First inference (Qwen3-0.6B)

**Goal:** Run the smallest possible vLLM inference inside the container. We use Qwen3-0.6B because it fits in memory in seconds, downloads fast, and is what we've already verified in earlier sessions.

**Steps:**

Create a tiny script on your laptop and `scp` it over, or just use a heredoc:

```bash
ssh ubuntu@<IP>
# on the VM
mkdir -p ~/d6
cat > ~/d6/smoke.py <<'PYEOF'
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    seed=42,
    enforce_eager=True,
    max_model_len=512,
    dtype="auto",
)
params = SamplingParams(temperature=0, max_tokens=20)
out = llm.generate(["The meaning of life is"], params)
gen = out[0].outputs[0]
# Print the int token IDs — that's our determinism check, not the decoded text
print("TOKEN_IDS:", list(gen.token_ids))
print("TEXT:", gen.text)
PYEOF

docker run --rm --gpus all --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/d6:/d6 \
  --entrypoint /bin/bash \
  ghcr.io/derpyplops/deterministic-serving:multinode \
  -c 'python3 /d6/smoke.py 2>&1 | tail -30' \
  | tee ~/d6/smoke-run-1.log
```

**Verification:**
- The script prints a `TOKEN_IDS:` line.
- The TOKEN_IDS are int values in `[0, vocab_size)`.
- The text decodes to something English-ish.

If you see CUDA OOM with this model on an 80GB H100, something is very wrong — log it and stop.

**Log entry:**
```markdown
### <timestamp> — MILESTONE: first inference on Lambda H100

Model: Qwen/Qwen3-0.6B
Wall time (engine init): ~8s
Wall time (1 prompt × 20 tokens): ~1s
TOKEN_IDS (first 5): <copy from log>
TEXT: <copy from log>
```

**Commit:**
```bash
# Copy the smoke script into the repo so future-you can find it
mkdir -p scripts/d6
scp ubuntu@<IP>:~/d6/smoke.py scripts/d6/phase1_smoke.py
git add scripts/d6/phase1_smoke.py experiments/d6-lambda-rollout-log.md
git commit -m "d6: phase 1 task 1.5 — first inference on Lambda H100"
```

---

### Task 1.6: Determinism repeat (the actual point of Phase 1)

**Goal:** Run the same script twice and prove the TOKEN_IDS are bitwise identical. This is the most basic determinism check possible.

**Why this matters:** Phase 1 only proves "we can do anything" if it also proves "we can do the same thing twice." Without this check, you don't know whether your Phase 2 failures are network bugs or single-GPU bugs.

**Steps:**

```bash
ssh ubuntu@<IP>
# Run twice
docker run --rm --gpus all --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/d6:/d6 \
  --entrypoint /bin/bash \
  ghcr.io/derpyplops/deterministic-serving:multinode \
  -c 'python3 /d6/smoke.py 2>&1 | grep TOKEN_IDS' > ~/d6/run-a.txt

docker run --rm --gpus all --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/d6:/d6 \
  --entrypoint /bin/bash \
  ghcr.io/derpyplops/deterministic-serving:multinode \
  -c 'python3 /d6/smoke.py 2>&1 | grep TOKEN_IDS' > ~/d6/run-b.txt

cat ~/d6/run-a.txt
cat ~/d6/run-b.txt
diff ~/d6/run-a.txt ~/d6/run-b.txt && echo "DETERMINISTIC ✓" || echo "NONDETERMINISTIC ✗"
```

**Verification:**
- `diff` exits 0 (no output, then `DETERMINISTIC ✓`).

**If diff fails:** This is a hard stop. Do not proceed to Phase 2. Possible causes:
- Two different vLLM versions in some pip cache. Should not happen with the container, but check `pip3 show vllm` inside it.
- `enforce_eager=True` is being ignored because torch.compile cached something. Restart the container; we use `--rm` so this should be a non-issue.
- Driver/firmware nondeterminism. Note the Lambda host's specific CUDA driver version in the log and escalate.

**Log entry:**
```markdown
### <timestamp> — MILESTONE: single-GPU determinism verified on Lambda

Two consecutive runs of Qwen3-0.6B (same seed, same prompt, same container)
produced identical TOKEN_IDS. Bitwise identical (diff exit 0).
```

**Commit:**
```
git add experiments/d6-lambda-rollout-log.md
git commit -m "d6: phase 1 task 1.6 — single-GPU determinism repeat passes"
```

---

### Task 1.7: Negative test (proof the test can fail)

**Goal:** Make sure your determinism check is actually checking something. A test that always passes is worse than no test.

**Steps:** Change the prompt and confirm the comparison detects the difference. (Don't bother changing the seed — with `temperature=0` greedy decoding, the seed doesn't affect the output.)

```bash
ssh ubuntu@<IP>
sed 's|"The meaning of life is"|"The opposite of hot is"|' ~/d6/smoke.py > ~/d6/smoke-cold.py

docker run --rm --gpus all --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/d6:/d6 \
  --entrypoint /bin/bash \
  ghcr.io/derpyplops/deterministic-serving:multinode \
  -c 'python3 /d6/smoke-cold.py 2>&1 | grep TOKEN_IDS' > ~/d6/run-c.txt

diff ~/d6/run-a.txt ~/d6/run-c.txt && echo "TEST IS BROKEN" || echo "NEG TEST OK ✓"
```

**Verification:**
- `NEG TEST OK ✓` — different prompts produce different tokens. Your comparison can detect a difference.

**Log entry:**
```markdown
### <timestamp> — MILESTONE: negative test passes

Different prompts produce different TOKEN_IDS. The determinism check
distinguishes match from mismatch. The test can fail when it should.
```

**Commit:**
```
git add experiments/d6-lambda-rollout-log.md
git commit -m "d6: phase 1 task 1.7 — negative test for determinism check"
```

---

### Task 1.8: End of Phase 1 (decision point)

**Goal:** Decide whether to proceed to Phase 2 or terminate Node 1 and stop.

**Steps:** Check the log for any unresolved SETBACK entries. If there are any, do not proceed.

If all green, **leave Node 1 running** — we'll reuse it as the head node in Phase 2.

**Log entry:**
```markdown
### <timestamp> — END Phase 1: single-node smoke test complete

Container works on Lambda H100. Single-GPU inference is deterministic and
the determinism check has been negative-tested. Proceeding to Phase 2.

Node 1 (`<id>`) remains running and will become the Ray head node in Phase 2.
```

**Commit:**
```
git commit --allow-empty -m "d6: end phase 1 — single-node smoke complete"
```

(Empty commit is fine here; it marks the boundary. If you have other unstaged changes from the phase, stage them first.)

---

## Phase 2: Two-Node Ray Cluster + Anti-Cheat Verification

Goal: Add a second H100 on a different physical machine, form a Ray cluster across both, run a real distributed (**PP=2**) inference, and prove with multiple independent checks that the inference is *actually distributed*. The "actually distributed" part is the entire point of this phase. Total effort: ~90 minutes including the second poll.

**Why PP=2 and not TP=2** (matters): vLLM's runner only auto-pins the cross-node NCCL settings (`NCCL_NET=Socket`, `NCCL_P2P_DISABLE`, `NCCL_SHM_DISABLE`, `NCCL_BUFFSIZE`, `NCCL_SOCKET_IFNAME`) when `pipeline_parallel_size > 1` *or* when `VLLM_MULTI_NODE=1` is set in the environment. See `cmd/runner/vllm_runner.py` lines 23–33. Phase 2 with PP=2 hits the `pipeline_parallel_size > 1` path naturally, which is what Phase 3 will use, so the NCCL config under test is the same. Using TP=2 here would either (a) require remembering to set `VLLM_MULTI_NODE=1` everywhere or (b) leave parts of the NCCL config un-pinned, which would invalidate Phase 2 as a Phase 3 dress rehearsal. Pick the workload that exercises the same code path.

### Task 2.1: Poll for the second H100

**Goal:** Get a second H100 up. **Same instance type and same region as Node 1 if possible** — keeps the GPUs identical and unlocks the rsync download strategy in Phase 3.

**Steps:**
```bash
# Look up Node 1's type AND region
NODE1_TYPE=$(python3 scripts/lambda/lambda_cli.py list | head -1 | awk '{print $2}')
# The list command doesn't show region directly; pull it from the API:
NODE1_REGION=$(curl -sf -u "$LAMBDALABS_API_KEY:" https://cloud.lambdalabs.com/api/v1/instances \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['region']['name'])")
echo "Node 1 type=$NODE1_TYPE region=$NODE1_REGION"

# Try same region first
python3 scripts/lambda/lambda_cli.py poll $NODE1_TYPE --count 1 --interval 30 --name-prefix d6-node-2 --region $NODE1_REGION

# If nothing for several minutes, drop the region pin
python3 scripts/lambda/lambda_cli.py poll $NODE1_TYPE --count 1 --interval 30 --name-prefix d6-node-2
```

If you can't get the same type, try a sibling (`gpu_1x_h100_pcie` instead of `_sxm5`). Note the divergence in the log; we may need to repeat Phase 2 if PCIe vs SXM produces different results.

When you see `launched`, ^C and confirm:
```bash
python3 scripts/lambda/lambda_cli.py list
# Should show 2 instances now
```

**Log entry:**
```markdown
### <timestamp> — COST: launched 2nd H100

Total instances: 2
Cost since Phase 1: $<rate × wall time>
```

---

### Task 2.2: Wait for Node 2 to be SSH-ready

Same as Task 1.2 but for the second instance. SSH in, verify GPU.

```bash
ssh ubuntu@<NODE2_IP> "nvidia-smi --query-gpu=name --format=csv,noheader"
```

---

### Task 2.3: Pull the container on Node 2

```bash
ssh ubuntu@<NODE2_IP> "docker pull ghcr.io/derpyplops/deterministic-serving:multinode"
ssh ubuntu@<NODE2_IP> "docker images ghcr.io/derpyplops/deterministic-serving:multinode"
```

**Log entry:**
```markdown
### <timestamp> — MILESTONE: container pulled on Node 2
```

---

### Task 2.4: Verify network connectivity between Node 1 and Node 2

**Goal:** Cross-node Ray and NCCL need bidirectional TCP. Verify it works on a high port *before* you waste time debugging Ray failures.

**Steps:**
```bash
# On Node 2, start a netcat listener
ssh ubuntu@<NODE2_IP> "nc -l -p 29500 &"

# From Node 1, try to connect
ssh ubuntu@<NODE1_IP> "nc -zv -w5 <NODE2_IP> 29500 && echo OK || echo FAIL"

# Other direction
ssh ubuntu@<NODE1_IP> "nc -l -p 29500 &"
ssh ubuntu@<NODE2_IP> "nc -zv -w5 <NODE1_IP> 29500 && echo OK || echo FAIL"

# Kill any lingering listeners
ssh ubuntu@<NODE1_IP> "pkill nc || true"
ssh ubuntu@<NODE2_IP> "pkill nc || true"
```

**Verification:**
- Both directions print `OK`.

**If it fails:** Lambda VMs allow all ports inbound by default; if a firewall is blocking you, it's likely on the VM itself (`ufw`) or on a network-level ACL. Check `sudo iptables -L INPUT` and `sudo ufw status`.

**Log entry:**
```markdown
### <timestamp> — MILESTONE: cross-node TCP works on port 29500

Both directions verified with nc. Node 1 ⇄ Node 2 reachable.
```

**Commit:** none.

---

### Task 2.5: Identify the network interface name on each node

**Goal:** NCCL needs to be told which NIC to use via `NCCL_SOCKET_IFNAME`. Lambda VMs do **not** always expose `eth0` — they may use `enp1s0`, `ens3`, etc. Both `vllm_runner.py` and Appendix B default to `eth0`, but that's a default, not a fact. Find the real name now.

**Steps:**
```bash
ssh ubuntu@<NODE1_IP> "ip -br link show | grep -v LOOPBACK | awk '{print \$1}'"
ssh ubuntu@<NODE2_IP> "ip -br link show | grep -v LOOPBACK | awk '{print \$1}'"
```

The first non-loopback line is your primary interface. Common values: `eth0`, `enp1s0`, `ens3`, `eno1`. Both nodes usually have the same name on Lambda but not always — check both.

**Save the value(s) to your shell** (you'll use them in 2.6 and beyond):

```bash
export NCCL_IFNAME=<the name you found, e.g. enp1s0>
```

If the two nodes have *different* interface names, that's still fine — we'll set `NCCL_SOCKET_IFNAME` per node when we launch containers. But log it.

**Verification:**
```bash
ssh ubuntu@<NODE1_IP> "ip -4 addr show $NCCL_IFNAME | grep inet"
# Should print an IP that matches the node's primary IP
```

**Log entry:**
```markdown
### <timestamp> — Identified NCCL interface

Node 1: $NCCL_IFNAME (IP <NODE1_IP>)
Node 2: $NCCL_IFNAME (IP <NODE2_IP>)
```

**Commit:** none.

---

### Task 2.6: Start the Ray cluster

**Goal:** Two nodes forming a Ray cluster with 2 GPUs total.

**The setup we'll use:**
- Node 1 = head (we already have it from Phase 1)
- Node 2 = worker
- Both run the container with `--network host`
- Ray binds to the host's network, not Docker's bridge
- We pass `NCCL_SOCKET_IFNAME` and `VLLM_MULTI_NODE=1` into the container so subsequent inference picks them up

**Steps:**

On **Node 1** (head), start Ray inside a long-running container. Don't use `--rm` here; we want it to stick around for multiple `docker exec`.

```bash
ssh ubuntu@<NODE1_IP>

# Get Node 1's primary IP (the one Node 2 will reach us at)
HEAD_IP=$(hostname -I | awk '{print $1}')
echo "HEAD_IP=$HEAD_IP"

# Launch the head container
docker run -d --name ray-head \
  --gpus all \
  --network host \
  --shm-size 10.24g \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/d6:/d6 \
  -e VLLM_MULTI_NODE=1 \
  -e NCCL_SOCKET_IFNAME=$NCCL_IFNAME \
  --entrypoint /bin/bash \
  ghcr.io/derpyplops/deterministic-serving:multinode \
  -c "ray start --head --node-ip-address=$HEAD_IP --port=6379 --num-gpus=1 --block"

# Wait a few seconds for Ray to come up
sleep 5
docker logs ray-head | tail -20
```

On **Node 2** (worker):
```bash
ssh ubuntu@<NODE2_IP>

WORKER_IP=$(hostname -I | awk '{print $1}')
HEAD_IP=<copy from above>
NCCL_IFNAME=<the name from Task 2.5>

docker run -d --name ray-worker \
  --gpus all \
  --network host \
  --shm-size 10.24g \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/d6:/d6 \
  -e VLLM_MULTI_NODE=1 \
  -e NCCL_SOCKET_IFNAME=$NCCL_IFNAME \
  --entrypoint /bin/bash \
  ghcr.io/derpyplops/deterministic-serving:multinode \
  -c "ray start --address=$HEAD_IP:6379 --node-ip-address=$WORKER_IP --num-gpus=1 --block"

sleep 5
docker logs ray-worker | tail -20
```

Back on **Node 1**:
```bash
docker exec ray-head ray status
```

**Verification:** `ray status` should report 2 nodes and 2 GPUs.

```
Node status
---------------------------------------------------------------
Active:
 1 node_<hash1>
 1 node_<hash2>

Resources
---------------------------------------------------------------
Usage:
 0.0/2.0 CPU
 0.0/2.0 GPU
 ...
```

**If it fails:**
- "Could not connect to Ray head": worker can't reach Node 1. Re-do Task 2.4 with port 6379.
- "GPU count is 0": `--gpus all` isn't propagating. `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` to verify the host's docker can see GPUs.

**Log entry:**
```markdown
### <timestamp> — MILESTONE: Ray cluster formed (2 nodes, 2 GPUs)

Head: <NODE1_IP>
Worker: <NODE2_IP>
Container: ghcr.io/derpyplops/deterministic-serving:multinode
ray status output:
<copy verbatim>
```

**Commit:** none.

---

### Task 2.7: Run a PP=2 distributed inference

**Goal:** Use vLLM with `pipeline_parallel_size=2` and `distributed_executor_backend="ray"` to split the model's *layers* across both GPUs (Node 1 holds the first half of layers, Node 2 holds the second half).

**Model choice:** Qwen3-0.6B is fine. PP=2 will split its ~24 transformer layers ~12-and-12 across the two nodes. We're not testing model size yet, just whether vLLM is exercising both nodes via NCCL.

**Why PP and not TP, again:** PP=2 triggers the `pipeline_parallel_size > 1` branch in `cmd/runner/vllm_runner.py:_set_deterministic_env`, which forces `NCCL_NET=Socket` and friends. That's the code path Phase 3 will use. We want Phase 2 to test the same path.

**Steps:**

On Node 1 (head):
```bash
ssh ubuntu@<NODE1_IP>

# Drop into the head container
docker exec -it ray-head bash

# Now inside the container — confirm the env vars from `docker run` are present
echo "VLLM_MULTI_NODE=$VLLM_MULTI_NODE  NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"

cat > /d6/pp2_smoke.py <<'PYEOF'
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    seed=42,
    enforce_eager=True,
    max_model_len=512,
    pipeline_parallel_size=2,
    tensor_parallel_size=1,
    distributed_executor_backend="ray",
)
params = SamplingParams(temperature=0, max_tokens=20)
out = llm.generate(["The meaning of life is"], params)
gen = out[0].outputs[0]
print("TOKEN_IDS:", list(gen.token_ids))
print("TEXT:", gen.text)
PYEOF

python3 /d6/pp2_smoke.py 2>&1 | tee /d6/pp2_smoke.log
```

**Verification:**
- The script prints `TOKEN_IDS:` and `TEXT:`.
- vLLM logs mention `pipeline_parallel_size=2` and `distributed_executor_backend=ray`.
- During execution, **both** nodes show GPU activity. (We'll verify this hard in Task 2.8.)

**Common failure here:** vLLM may complain that the layer count doesn't divide evenly by `pp_size`, or that PP requires more than one Ray actor. Read the error and re-acquire if needed. If Ray placed both ranks on one node, `ray status` will tell you (you'll see one node with 2/1 GPU usage and one idle).

**Log entry:**
```markdown
### <timestamp> — MILESTONE: PP=2 inference completes via Ray cluster

Model: Qwen/Qwen3-0.6B (pp=2)
Wall time: <seconds>
TOKEN_IDS: <copy>
```

**Commit:** none yet — Task 2.8 is the real verification.

---

### Task 2.8: Anti-cheat verification (the actual point of Phase 2)

**Goal:** Prove that the PP=2 run *actually* distributed work across the two physical nodes. The failure mode we're ruling out: Ray scheduled both pipeline stages on the same node (or vLLM silently fell back to single-GPU), the second node sat idle, and you congratulated yourself for nothing.

We will run **four independent checks**, each of which would fail if the run were not really distributed.

#### Check A: Both GPUs hold their share of the model layers

A PP=2 run splits the model's transformer layers across both GPUs — Node 1 holds the first half, Node 2 holds the second half. Each GPU should show non-zero memory used during inference. If one GPU has all the layers (full weight footprint) and the other has none, PP isn't real.

For Qwen3-0.6B with PP=2, expect each GPU to hold roughly half of the weight memory plus its own KV cache slice. The two halves don't have to be exactly equal (the embedding layer and lm_head live on the boundaries) but neither should be zero.

**Steps:**

Run a longer inference (many requests, many tokens) so you have a steady-state window to read both nvidia-smi outputs:

```python
# /d6/pp2_long.py
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3-0.6B", seed=42, enforce_eager=True,
          max_model_len=1024, pipeline_parallel_size=2, tensor_parallel_size=1,
          distributed_executor_backend="ray")
params = SamplingParams(temperature=0, max_tokens=512)
out = llm.generate(["Tell me a long story about " + str(i) for i in range(8)], params)
for o in out:
    print(o.outputs[0].text[:60])
```

Run it on Node 1 (`docker exec ray-head python3 /d6/pp2_long.py`), then in two other terminals query GPU memory on each node:

```bash
# Terminal 2 — Node 1
ssh ubuntu@<NODE1_IP> "while true; do nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; sleep 1; done"

# Terminal 3 — Node 2
ssh ubuntu@<NODE2_IP> "while true; do nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; sleep 1; done"
```

**Verification:**
- Both GPUs show memory used (>0) during inference. Compare the *peak* memory, not just one snapshot.
- Memory used per GPU is in the same order of magnitude (don't expect exact 50/50 — embeddings and lm_head live at the pipeline boundaries).
- If Node 2's GPU shows 0 MiB throughout the inference: PP is not distributing. **Stop and diagnose.** Check `ray status` for the actual placement.

**Log entry:**
```markdown
### <timestamp> — Check A: Both GPUs in use

Node 1 GPU peak memory during PP=2 inference: <X> MiB / 81920 MiB
Node 2 GPU peak memory during PP=2 inference: <Y> MiB / 81920 MiB
Both > 0: PASS
```

#### Check B: NCCL traffic is observed in logs

A real PP=2 run does point-to-point send/recv between the two pipeline stages on every forward pass. With `NCCL_DEBUG=INFO`, NCCL prints which IPs and interfaces it's communicating over.

**Steps:**

```bash
# Inside the head container, re-run with NCCL_DEBUG=INFO
docker exec -it ray-head bash
NCCL_DEBUG=INFO python3 /d6/pp2_smoke.py 2>&1 | tee /d6/nccl_debug.log
grep -i 'NCCL INFO' /d6/nccl_debug.log > /d6/nccl_info_only.log

# Look for the IP addresses in the NCCL logs
grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' /d6/nccl_info_only.log | sort -u
```

**Verification:**
- The NCCL log mentions both Node 1's IP and Node 2's IP.
- You see lines containing `via NET/Socket/<n>` (confirms NCCL is using the TCP socket transport, not falling back to something else).
- You see `Init COMPLETE` for the comm.
- You see `NCCL INFO comm 0x... rank 0 nranks 2 ...` and a corresponding rank 1 line on the other node.

If the NCCL log only shows local-loopback (`127.0.0.1`) addresses or only a single IP: both ranks landed on the same node. Ray didn't schedule the second pipeline stage on Node 2. Run `docker exec ray-head ray status` and look at GPU usage per node — one will show 2/1 and the other 0/1.

**Log entry:**
```markdown
### <timestamp> — Check B: NCCL cross-node traffic confirmed

NCCL_DEBUG=INFO output mentions IPs: <list>
Ring topology: <copy a few of the "Ring N : ... via NET/Socket" lines>
```

#### Check C: A negative network test breaks the run

This is the strongest check. Block traffic between nodes and run a *fresh* inference. It must fail.

**Why we run a fresh inference (not block mid-flight):** `iptables -A INPUT -j DROP` only blocks new packets. NCCL's existing TCP sockets, established before the rule was inserted, will keep flowing on some kernels because the conntrack entries are still valid. Two ways to do this safely:

1. **Insert the rule first, then start a brand-new vLLM process.** It has to establish new NCCL sockets, which the rule will block. *Cleanest.*
2. **Use `-j REJECT --reject-with tcp-reset` instead of `DROP`** — sends RSTs that close existing connections too. Then `conntrack -D -d <IP>` to flush the table for paranoia.

We use approach 1.

**Steps:**

Stop any vLLM processes running inside the container first:
```bash
docker exec ray-head pkill -f 'python3 /d6/pp2' || true
```

Insert the iptables rules on **both** nodes (block in both directions). Use `REJECT` so any stale connections are torn down too:

```bash
ssh ubuntu@<NODE2_IP> "
  sudo iptables -I INPUT 1 -s <NODE1_IP> -j REJECT --reject-with tcp-reset
  sudo iptables -I OUTPUT 1 -d <NODE1_IP> -j REJECT --reject-with tcp-reset
  sudo iptables -L INPUT --line-numbers | head -5
"
ssh ubuntu@<NODE1_IP> "
  sudo iptables -I INPUT 1 -s <NODE2_IP> -j REJECT --reject-with tcp-reset
  sudo iptables -I OUTPUT 1 -d <NODE2_IP> -j REJECT --reject-with tcp-reset
"
```

Confirm the block actually works at the TCP level before running inference:
```bash
ssh ubuntu@<NODE1_IP> "timeout 5 nc -zv <NODE2_IP> 6379 2>&1 || echo BLOCKED"
# Expected: 'BLOCKED' (or "Connection refused/timed out")
```

Now run a *fresh* PP=2 inference. The Ray cluster will still appear up (Ray's heartbeats may keep working briefly via existing sockets, but new NCCL traffic will be blocked):

```bash
ssh ubuntu@<NODE1_IP>
timeout 180 docker exec ray-head python3 /d6/pp2_smoke.py 2>&1 | tee /d6/pp2_blocked.log
echo "exit code: $?"
```

**CRITICAL CLEANUP** (do this even if the previous step is still hanging — kill it first):
```bash
ssh ubuntu@<NODE2_IP> "
  sudo iptables -D INPUT -s <NODE1_IP> -j REJECT --reject-with tcp-reset
  sudo iptables -D OUTPUT -d <NODE1_IP> -j REJECT --reject-with tcp-reset
  sudo iptables -L INPUT --line-numbers
"
ssh ubuntu@<NODE1_IP> "
  sudo iptables -D INPUT -s <NODE2_IP> -j REJECT --reject-with tcp-reset
  sudo iptables -D OUTPUT -d <NODE2_IP> -j REJECT --reject-with tcp-reset
"
# Confirm cleanup worked
ssh ubuntu@<NODE1_IP> "nc -zv <NODE2_IP> 6379 && echo OK"
```

**Verification:**
- The blocked inference exited non-zero (timeout 124, or a NCCL/Ray connection error). Anything except a successful run with `TOKEN_IDS:` output.
- After cleanup, `nc -zv` between nodes succeeds.

**If the blocked inference actually finished with TOKEN_IDS output:** distribution is fake — both ranks are running on the same node. That's a hard fail; do not proceed to Phase 3 until you understand why.

**Note on Ray fault tolerance:** Ray *can* be configured to retry failed actors on other nodes. vLLM does not enable that for inference workers, but if your inference somehow completes despite the network block, double-check `ray status` and the logs to confirm Ray didn't transparently reschedule. If it did, that's a Ray config issue and Phase 2 needs `--max-restarts 0` somewhere. Log it and ask.

**Log entry:**
```markdown
### <timestamp> — Check C: Negative network test passes

Inserted REJECT --reject-with tcp-reset on Node 1 ↔ Node 2 in both directions.
Verified TCP block with nc.
PP=2 inference attempted from Node 1.
Result: <exit code, error message verbatim, e.g. "exit 124, NCCL Init failed">
Restored iptables. Verified Node 1 ↔ Node 2 reachable again.
```

#### Check D: A model that doesn't fit on one GPU

The most ironclad check: run a model that's larger than 80 GiB. If PP=2 isn't real, it won't fit on a single GPU.

For the Phase 2 check, this is overkill. We're going to do it implicitly in Phase 3 with Mistral Large 2 (~240 GB, requires PP=4 or TP=4). Note this in the log as deferred.

**Log entry:**
```markdown
### <timestamp> — Check D: deferred to Phase 3

Will rely on Mistral Large 2 (~240 GB) and DBRX (~265 GB) each exceeding
the aggregate VRAM of 1×H100 (80 GB) and 2×H100 (160 GB) to provide the
implicit "must distribute" check in Phase 3. Neither model can fit on
fewer than 4 GPUs at bf16, so the mere fact that they complete inference
is the structural proof that 4-way distribution is real.
```

#### Phase 2 verification summary

```markdown
### <timestamp> — MILESTONE: distributed inference is real (Phase 2 anti-cheat passes)

| Check | Result |
|-------|--------|
| A: Both GPUs use memory | PASS (Node 1: <X> MiB, Node 2: <Y> MiB) |
| B: NCCL cross-node traffic in logs | PASS (saw both IPs in NCCL INFO) |
| C: iptables block breaks inference | PASS (timed out / errored) |
| D: Oversized model | DEFERRED to Phase 3 |
```

**Commit:**
```bash
# Save the verification scripts so they're reproducible
mkdir -p scripts/d6
# scp pp2_smoke.py from Node 1 into scripts/d6/
scp ubuntu@<NODE1_IP>:~/d6/pp2_smoke.py scripts/d6/phase2_pp2_smoke.py
git add scripts/d6/*.py experiments/d6-lambda-rollout-log.md
git commit -m "d6: phase 2 task 2.8 — anti-cheat verification of distributed inference"
```

---

### Task 2.9: Determinism check on PP=2

**Goal:** Now that we know the inference is really distributed, verify two PP=2 runs produce the same TOKEN_IDS.

**Steps:**

```bash
# On Node 1, inside the head container
docker exec ray-head python3 /d6/pp2_smoke.py 2>&1 | grep TOKEN_IDS > ~/d6/pp2_run_a.txt
docker exec ray-head python3 /d6/pp2_smoke.py 2>&1 | grep TOKEN_IDS > ~/d6/pp2_run_b.txt
diff ~/d6/pp2_run_a.txt ~/d6/pp2_run_b.txt && echo "DETERMINISTIC ✓" || echo "NONDETERMINISTIC ✗"
```

**Verification:** `DETERMINISTIC ✓`.

**If nondeterministic:**
- This is the failure D6 is *trying to detect.* Don't paper over it.
- Check that the runner's NCCL pinning actually fired. Inside the container, before launching another inference:
  ```bash
  docker exec ray-head env | grep -E '^NCCL_|^VLLM_MULTI'
  ```
  You should see `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_NET=Socket`, `NCCL_SOCKET_IFNAME`, `NCCL_P2P_DISABLE`, `NCCL_SHM_DISABLE`, `NCCL_BUFFSIZE`, and `VLLM_MULTI_NODE=1`. If anything is missing, the `-e` flags on `docker run` didn't propagate, or the smoke script imports vllm before the runner sets them. Source of truth: `cmd/runner/vllm_runner.py:_set_deterministic_env` lines 15–36.
- If they're all set and the run is still nondeterministic, this is a real D6 finding. Do not paper it over. Log it and stop.

**Log entry:**
```markdown
### <timestamp> — MILESTONE: PP=2 over Ray cluster is deterministic

Two consecutive PP=2 runs of Qwen3-0.6B produced identical TOKEN_IDS.
NCCL env active in container: <copy the env|grep output>
```

**Commit:**
```
git add experiments/d6-lambda-rollout-log.md
git commit -m "d6: phase 2 task 2.9 — PP=2 over Ray cluster is deterministic"
```

---

### Task 2.10: End of Phase 2

**Steps:** Decide whether to keep both nodes running for Phase 3 (yes if you're going straight into it) or terminate and re-acquire later.

If you're proceeding immediately, keep both running. The Ray cluster will be expanded in Phase 3.

**Log entry:**
```markdown
### <timestamp> — END Phase 2: distributed inference works and is deterministic

Both nodes still running. Cluster is alive. Anti-cheat checks all green.
Proceeding to Phase 3.
```

**Commit:**
```
git commit --allow-empty -m "d6: end phase 2 — distributed inference verified deterministic"
```

---

## Phase 3: Full 4-Node D6 Experiment

Goal: Add two more H100s, run the full D6 test harness against Mistral Large 2 (dense) and DBRX (MoE), write the report. Both models are large enough that they cannot fit on 1 or 2 H100s at bf16 — this is the structural proof that the 4-GPU coordination is actually doing work that 1–2 GPUs could not. Total effort: ~3–5 hours including model downloads.

### Task 3.1: Poll for the remaining 2 H100s

**Same type, same region** as Nodes 1 and 2 if possible. Two more nodes, not one.

```bash
NODE1_REGION=$(curl -sf -u "$LAMBDALABS_API_KEY:" https://cloud.lambdalabs.com/api/v1/instances \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['region']['name'])")

# Same region first
python3 scripts/lambda/lambda_cli.py poll <SAME_TYPE> --count 2 --interval 30 \
  --name-prefix d6-node-3 --region $NODE1_REGION

# If nothing for several minutes, drop the region pin (rsync strategy in 3.7 will be unavailable)
python3 scripts/lambda/lambda_cli.py poll <SAME_TYPE> --count 2 --interval 30 --name-prefix d6-node-3

python3 scripts/lambda/lambda_cli.py list  # should show 4 instances now
```

**Log entry:**
```markdown
### <timestamp> — COST: launched 3rd and 4th H100

Total nodes: 4.
Cumulative cost so far: $<rate × wall>
```

---

### Task 3.2: Wait for both to be SSH-ready

Same procedure as Task 1.2, twice. Pull the container on both:

```bash
ssh ubuntu@<NODE3_IP> "docker pull ghcr.io/derpyplops/deterministic-serving:multinode"
ssh ubuntu@<NODE4_IP> "docker pull ghcr.io/derpyplops/deterministic-serving:multinode"
```

---

### Task 3.3: Verify cross-node connectivity for all pairs

For each pair (1↔3, 1↔4, 2↔3, 2↔4, 3↔4), run the netcat connectivity check from Task 2.4. Six pairs total.

**Log entry:**
```markdown
### <timestamp> — MILESTONE: all 6 cross-node TCP pairs verified
```

---

### Task 3.4: Add Nodes 3 and 4 to the Ray cluster

On each new worker. **Re-do Task 2.5 (interface name) on Nodes 3 and 4 first** — they may have a different interface name than Nodes 1 and 2.

```bash
ssh ubuntu@<NODE3_IP>

WORKER_IP=$(hostname -I | awk '{print $1}')
HEAD_IP=<NODE1_IP>
NCCL_IFNAME=<the name from `ip -br link` on this node>

docker run -d --name ray-worker \
  --gpus all \
  --network host \
  --shm-size 10.24g \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/d6:/d6 \
  -e VLLM_MULTI_NODE=1 \
  -e NCCL_SOCKET_IFNAME=$NCCL_IFNAME \
  --entrypoint /bin/bash \
  ghcr.io/derpyplops/deterministic-serving:multinode \
  -c "ray start --address=$HEAD_IP:6379 --node-ip-address=$WORKER_IP --num-gpus=1 --block"
```

Same for Node 4.

Back on Node 1:
```bash
docker exec ray-head ray status
# Should show 4 nodes, 4 GPUs
```

**Log entry:**
```markdown
### <timestamp> — MILESTONE: Ray cluster has 4 nodes, 4 GPUs

ray status output:
<copy verbatim>
```

---

### Task 3.5: Generate lockfiles for the multinode manifests

**Goal:** The runner requires `--lockfile` alongside `--manifest`. Lockfiles for our four new multinode manifests don't exist yet. Generate them.

**There is no escape hatch.** The runner enforces:
- `lockfile['manifest_digest'] == sha256(canonical_json(manifest))` (rejection if mismatch)
- `lockfile['canonicalization']['lockfile_digest'] == sha256(canonical_json(lockfile))` (rejection if mismatch)

So you cannot copy a lockfile from a different manifest and edit it. You also cannot copy the same model's existing lockfile (e.g. `mistral-large2-tp4` → `mistral-large2-tp4-multinode`) because the manifest digest is over the entire manifest including the runtime block, which differs between the multinode and original variants.

**The right tool:** `cmd/resolver/main.py`. CLI:

```
python3 cmd/resolver/main.py \
  --manifest manifests/<name>.manifest.json \
  --lockfile-out lockfiles/<name>.lockfile.json \
  --resolve-hf \
  --hf-token-file ~/.hf-token
```

**Steps:**

1. Confirm `lockfiles/` exists; create it if needed:
   ```bash
   mkdir -p lockfiles
   ```
2. Get a HuggingFace token. Mistral Large 2 is gated, so your account must have been approved for it. DBRX (`databricks/dbrx-instruct`) is **open-access** — no gating, but you still need a token to use the HF API. Save the token to `~/.hf-token` and `chmod 600` it. (Or use `--hf-token <token>` directly, but don't paste tokens into shell history.)
3. Run the resolver for each of the four manifests. For the DBRX manifests, which are stubs, pass `--manifest-out` so the resolved HF weight shards are written back into the manifest file on disk:
   ```bash
   for m in mistral-large2-pp4-multinode mistral-large2-tp4-multinode; do
     python3 cmd/resolver/main.py \
       --manifest manifests/${m}.manifest.json \
       --lockfile-out lockfiles/${m}.lockfile.json \
       --resolve-hf \
       --hf-token-file ~/.hf-token
   done

   for m in dbrx-pp4-multinode dbrx-tp4-multinode; do
     python3 cmd/resolver/main.py \
       --manifest manifests/${m}.manifest.json \
       --manifest-out manifests/${m}.manifest.json \
       --lockfile-out lockfiles/${m}.lockfile.json \
       --resolve-hf \
       --hf-token-file ~/.hf-token
   done
   ```

   The resolver hits HF Hub for each model, reads the file index, computes/verifies digests, and writes a lockfile. Expect 1–3 minutes per manifest depending on HF latency.

4. Skim each generated lockfile to confirm it has artifacts:
   ```bash
   python3 -c "
   import json, sys
   for p in sys.argv[1:]:
     d = json.load(open(p))
     print(f'{p}: {len(d[\"artifacts\"])} artifacts, manifest_digest={d[\"manifest_digest\"][:16]}...')
   " lockfiles/*-multinode.lockfile.json
   ```

**Verification:**
- Four lockfile files exist in `lockfiles/`.
- Each lockfile validates against the schema:
  ```bash
  for f in lockfiles/*-multinode.lockfile.json; do
    python3 cmd/runner/main.py --manifest manifests/$(basename $f .lockfile.json).manifest.json \
      --lockfile $f --out-dir /tmp/d6-validate-only --mode synthetic --replica-id replica-0 \
      --validate-only 2>&1 | tail -3 || echo "FAIL: $f"
  done
  ```
  (If `--validate-only` doesn't exist on the runner, drop it — `--mode synthetic` is the lightest path that exercises validation without launching vLLM.)

**If the resolver fails:**
- HF gating: `403 Forbidden` → request access on the model card and wait. Don't bypass.
- Network: timeouts → retry. The resolver is idempotent.
- Schema mismatch: the manifest may be missing required fields; compare to `schemas/manifest.v1.schema.json` and fix the manifest.

**Do not commit `~/.hf-token` or any file containing the raw token.**

**Log entry:**
```markdown
### <timestamp> — MILESTONE: lockfiles generated for all 4 multinode manifests

Generator: cmd/resolver/main.py --resolve-hf
Files:
  - lockfiles/mistral-large2-pp4-multinode.lockfile.json (<N> artifacts)
  - lockfiles/mistral-large2-tp4-multinode.lockfile.json (<N> artifacts)
  - lockfiles/dbrx-pp4-multinode.lockfile.json           (<N> artifacts)
  - lockfiles/dbrx-tp4-multinode.lockfile.json           (<N> artifacts)
```

**Commit:**
```
git add lockfiles/*-multinode.lockfile.json experiments/d6-lambda-rollout-log.md
git commit -m "d6: phase 3 task 3.5 — generate lockfiles for multinode manifests"
```

---

### Task 3.6: Push manifests and runner code to all 4 nodes

The container has the runner code baked in (from when it was built). But the **multinode manifests we created in this session may not be in the container**. Confirm:

```bash
ssh ubuntu@<NODE1_IP>
docker exec ray-head ls /workspace/manifests/ | grep multinode
```

If the multinode manifests are missing, they need to be mounted into the container. Either:

1. **scp the manifests to each node's `~/d6/manifests/`** and bind-mount that directory:
```bash
mkdir -p ~/d6/manifests
# from your laptop:
scp manifests/*multinode*.manifest.json ubuntu@<NODE1_IP>:~/d6/manifests/
# repeat for nodes 2-4
# then add `-v ~/d6/manifests:/workspace/manifests-overlay` to your docker run args
# and reference manifests via that path
```

2. **Pull the latest branch into the container** by exec-ing in and running `git pull` if the workdir is a git repo.

Pick whichever is faster for you. Log it.

---

### Task 3.7: Download model weights to all 4 nodes

**Goal:** Each Lambda VM needs a local copy of Mistral Large 2 (~240 GB) and DBRX (~265 GB). This is the longest non-compute step. Plan for **45–90 minutes per model per node**, depending on HF bandwidth and concurrency. DBRX is the larger of the two and will dominate this step.

**Math you should care about:**
- Mistral Large 2: ~240 GB × 4 nodes = ~1 TB aggregate from HF.
- DBRX: ~265 GB × 4 nodes = ~1.1 TB aggregate from HF.
- Total disk per node: ~505 GB. Confirm `df -h ~/.cache/huggingface` has room before you start. Lambda VMs typically come with 1+ TB, but if your `--disk` is smaller than ~600 GB you don't have enough margin — re-launch with more.
- HF rate-limits aggressive parallel pulls. Don't spawn 16 connections per node.
- **Both models need ~265 GB of VRAM spread across 4×H100 (320 GB total).** Neither fits on 1×H100 (80 GB) or 2×H100 (160 GB) at bf16. That's the point — the fact that Phase 3 inference completes at all is the structural proof that 4-way distribution is real.

**Strategy A — straight parallel download (simplest, works if all nodes have HF bandwidth):**

On each node, in parallel terminals:
```bash
ssh ubuntu@<NODE_IP>
# Run inside the container so we use the same HF lib version
docker exec -d ray-head bash -c '
export HF_TOKEN=<token>
huggingface-cli download mistralai/Mistral-Large-Instruct-2407 2>&1 | tee /d6/dl-mistral.log
huggingface-cli download databricks/dbrx-instruct 2>&1 | tee /d6/dl-dbrx.log
'
```

(For workers, exec into `ray-worker` instead of `ray-head`.) **Stagger by 60–120 seconds** between nodes so you don't all hit HF at the same instant; HF tends to rate-limit bursts more aggressively than steady streams.

**Strategy B — download once, rsync to peers (faster if all nodes are in the same Lambda region):**

Lambda's intra-region bandwidth (~10 Gbps) is much faster than HF's per-host throughput (~100 MB/s ≈ 0.8 Gbps). If you successfully pinned all 4 nodes to the same region in Tasks 1.1, 2.1, 3.1, you can save substantial time:

1. Download both models to **Node 1 only**:
   ```bash
   ssh ubuntu@<NODE1_IP>
   docker exec ray-head bash -c '
     export HF_TOKEN=<token>
     huggingface-cli download mistralai/Mistral-Large-Instruct-2407
     huggingface-cli download databricks/dbrx-instruct
   '
   ```
2. Set up SSH from Node 1 to Nodes 2–4 (you may already have this from the experiment so far). On Node 1:
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/internal -N ""
   for ip in <NODE2_IP> <NODE3_IP> <NODE4_IP>; do
     ssh-copy-id -i ~/.ssh/internal.pub ubuntu@$ip
   done
   ```
3. rsync the HF cache from Node 1 to each peer in parallel:
   ```bash
   ssh ubuntu@<NODE1_IP> "
     for ip in <NODE2_IP> <NODE3_IP> <NODE4_IP>; do
       rsync -av --info=progress2 ~/.cache/huggingface/ ubuntu@\$ip:~/.cache/huggingface/ &
     done
     wait
   "
   ```

Strategy B trades complexity for speed. Use it only if all 4 nodes are confirmed same-region.

**Watch progress:**
```bash
for ip in <NODE1_IP> <NODE2_IP> <NODE3_IP> <NODE4_IP>; do
  echo -n "$ip: "
  ssh ubuntu@$ip "du -sh ~/.cache/huggingface/hub 2>/dev/null"
done
```

**Verification (after download, on every node):**
```bash
ssh ubuntu@<NODE_IP> "ls ~/.cache/huggingface/hub/models--mistralai--Mistral-Large-Instruct-2407/snapshots/*/ | wc -l"
# Expected: ~51 safetensor files + config + tokenizer
ssh ubuntu@<NODE_IP> "ls ~/.cache/huggingface/hub/models--databricks--dbrx-instruct/snapshots/*/ | wc -l"
# Expected: ~61 safetensor files + config + tokenizer + tiktoken.model
```

**Log entry:**
```markdown
### <timestamp> — MILESTONE: model weights present on all 4 nodes

Strategy used: A (direct from HF) | B (rsync from Node 1)
Mistral Large 2: ~240 GB per node, wall time <minutes>
DBRX:            ~265 GB per node, wall time <minutes>
```

**Commit:** none.

---

### Task 3.8: Run the existing D6 harness against the new manifests

**Goal:** Reuse `scripts/ci/d6_multinode_determinism.sh`. It already does the three D6 tests against a manifest you give it.

**Steps:** On Node 1, inside the head container:

```bash
docker exec -it ray-head bash

# Mistral Large 2 PP=4
MANIFEST=/workspace/manifests-overlay/mistral-large2-pp4-multinode.manifest.json \
TP_MANIFEST=/workspace/manifests-overlay/mistral-large2-tp4-multinode.manifest.json \
LOCKFILE=/workspace/lockfiles/mistral-large2.lockfile.json \
OUT_DIR=/tmp/d6/mistral-large2 \
SKIP_TP=0 \
bash /workspace/scripts/ci/d6_multinode_determinism.sh
```

This runs three tests:
- PP=4 same-config (Run A vs Run A')
- PP=4 batch+order invariance (Run A vs shuffled Run B)
- TP=4-over-TCP same-config (the stretch goal)

When it finishes, all three should report `PASS`. If any FAIL, the harness writes a summary file in `$OUT_DIR/<test>_summary.json`.

Repeat for DBRX:
```bash
MANIFEST=/workspace/manifests-overlay/dbrx-pp4-multinode.manifest.json \
TP_MANIFEST=/workspace/manifests-overlay/dbrx-tp4-multinode.manifest.json \
LOCKFILE=/workspace/lockfiles/dbrx-pp4-multinode.lockfile.json \
OUT_DIR=/tmp/d6/dbrx \
SKIP_TP=0 \
bash /workspace/scripts/ci/d6_multinode_determinism.sh
```

**Verification:**
```bash
cat /tmp/d6/mistral-large2/*_summary.json
cat /tmp/d6/dbrx/*_summary.json
```

All `status` fields should be `PASS`.

**Log entry (per model):**
```markdown
### <timestamp> — D6 results: Mistral Large 2

| Test | Status | Match | Total Tokens |
|------|--------|-------|--------------|
| PP=4 same-config | PASS | 100/100 | <N> |
| PP=4 batch+order | PASS | 100/100 | <N> |
| TP=4-over-TCP same-config | PASS | 100/100 | <N> |

Wall time: <minutes> for all three tests.
```

---

### Task 3.9: Collect results

**Goal:** Pull the result directories back to your laptop so you can include them in the report.

```bash
mkdir -p experiments/multinode_determinism/$(date +%Y%m%d)/
scp -r ubuntu@<NODE1_IP>:/tmp/d6/* experiments/multinode_determinism/$(date +%Y%m%d)/
```

---

### Task 3.10: Write the D6 report

**Goal:** A markdown report at `experiments/D6_MULTINODE_DETERMINISM_REPORT.md` matching the structure of `experiments/MULTI_GPU_DETERMINISM_REPORT.md`.

The report should have:

1. **Summary** — a single sentence on whether D6 passes.
2. **Setup** — the cluster topology (4× H100, Lambda, IPs, regions, Ray version, vLLM version, NCCL settings).
3. **Models tested** — Mistral Large 2 (dense, ~123B params, ~240 GB bf16) and DBRX (MoE, 132B params, 16 experts top-4 routing, ~265 GB bf16), with HF revisions.
4. **Tests** — the three tests, in a table, with PASS/FAIL and request match counts.
5. **Cost** — total spend and breakdown.
6. **Conclusion** — what this proves about distributed determinism and what the next step would be.

Don't write more than 2 pages. Cite the experiment log for the gory details.

**Commit:**
```
git add experiments/D6_MULTINODE_DETERMINISM_REPORT.md \
        experiments/multinode_determinism/ \
        experiments/d6-lambda-rollout-log.md
git commit -m "d6: add D6 multinode determinism report and results"
```

---

### Task 3.11: Teardown

**Goal:** Stop the meter on all 4 instances. Be paranoid.

```bash
python3 scripts/lambda/lambda_cli.py terminate-all
# Type 'yes' to confirm

# Wait a few seconds, then verify nothing is running
python3 scripts/lambda/lambda_cli.py list
# Should print "(no instances)"
```

**Cross-check:** Open `https://cloud.lambda.ai/instances` in a browser (yes, even though the API says nothing is running). The web UI is the source of truth for billing.

**Log entry:**
```markdown
### <timestamp> — END Phase 3: D6 complete, all instances terminated

API list: 0 instances.
Web UI confirmed: 0 instances.
Total spend: $<...>
```

**Commit:**
```
git commit --allow-empty -m "d6: end phase 3 — D6 complete, instances terminated"
```

---

### Task 3.12: PR

Open a pull request from `multi-gpu-determinism` (or your branch) to `main` with:
- Title: `D6: Multi-node distributed determinism on Lambda`
- Body: link to `experiments/D6_MULTINODE_DETERMINISM_REPORT.md` and `experiments/d6-lambda-rollout-log.md`. List the four manifests added.

---

## Test Design Notes (Crash Course)

You may not need this section if you're already comfortable with test design. Read it if you've never written a test fixture or you're not sure why someone would care about negative tests.

### What makes a determinism test good

The test in this experiment is structurally simple: run twice, compare. But "compare" hides a lot of subtlety. A good determinism test:

1. **Compares the right thing.** For us, that's the integer token IDs, not the decoded text. Decoded text can be identical even when the tokens diverged (e.g., two different token sequences decoding to the same string), and tokens can diverge even when text is identical (different tokenizer outputs). Always compare the smallest, most stable representation.

2. **Has a negative test.** A test that always passes is a test that never tells you anything. Task 1.7 is this — change the prompt, confirm the comparison detects the difference. Without it, your "PASS" means nothing.

3. **Fixes everything that can be fixed.** Seed, eager mode, NCCL algo, batch size, prompt order, environment variables. Any free variable is a potential source of nondeterminism. The manifest declares them; the runner enforces them.

4. **Pins the hardware.** Manifests include a `hardware_profile`. The runner refuses to run on the wrong hardware unless `strict_hardware: false`. We set it false for D6 to allow any H100 SXM, but we log the actual hardware.

5. **Is small enough to run frequently and large enough to catch real bugs.** 100 prompts × 100 tokens is a sweet spot — enough to expose batching bugs, fast enough to iterate.

### What makes a bad determinism test

- Comparing decoded text instead of tokens.
- Using `assert "approximately equal"` for what should be bitwise equal.
- Running once and trusting it.
- Catching exceptions that should propagate.
- Adding retries that mask flakiness instead of fixing it.

### How to debug a failed determinism check

When two runs diverge, the question is: at which token did they diverge?

Look at `compare_runs` in `scripts/ci/d6_multinode_determinism.sh` — it reports the divergence point in the form `req-XX: DIVERGE at token N (A vs B)`. That tells you:
- Was it a single request or all of them?
- Was it the first token or after many?
- What was the divergence?

A divergence at token 0 means the model state diverged before any decoding, which usually points to weight loading or initialization. A divergence at token 50+ means accumulated rounding error from a non-pinned reduction.

### How to add a new test (if you need to)

Look at `tests/determinism/test_d4_batch_order_invariance.py` for the canonical pattern:
- It loads a fixture manifest.
- It calls the runner twice with different settings.
- It compares observables.
- It uses pytest fixtures (not unittest.setUp).

A test you might want to add for D6: `tests/determinism/test_d6_pp_invariance.py`. Skip it for the first pass — the shell harness covers the same ground.

---

## Troubleshooting

### "ray status" shows fewer nodes than expected

The worker container probably crashed. `docker logs ray-worker` on each worker node. Common causes:
- `--num-gpus=1` mismatch with what the host advertises. Check `nvidia-smi`.
- Head IP unreachable from the worker. Re-do Task 2.4 for that pair.
- Ray version mismatch — should not happen since both use the same image.

### vLLM hangs on init

Check `docker logs ray-head` for messages like `Waiting for X workers`. If it's stuck waiting:
- The Ray cluster is missing a worker or has a stale worker. `ray stop --force` on every node, then re-form the cluster.
- vLLM's HF cache needs to be warm — first-time model loads can take many minutes, especially for DBRX (~265 GB of weights to page in). Be patient and watch `du -sh ~/.cache/huggingface`.

### NCCL "unhandled cuda error"

Almost always means NCCL can't reach a peer. Re-do the netcat connectivity test on the NCCL ports (29500–29600 by default; we set NCCL_BUFFSIZE but not NCCL_PORT — vLLM picks one). Set `NCCL_DEBUG=INFO` in the env and re-run; the error message will name the peer it couldn't reach.

### Determinism check fails between two runs

This is the failure D6 is trying to detect. Don't paper over it. Investigate:
1. Are NCCL_ALGO and NCCL_PROTO actually pinned in the inference process? `python3 -c 'import os; print({k:v for k,v in os.environ.items() if k.startswith("NCCL")})'`
2. Are CUBLAS_WORKSPACE_CONFIG and friends set? Same check with `CUBLAS`/`CUDA`.
3. Is `enforce_eager=True` actually being honored? Check the vLLM init logs for "CUDA Graphs" — they should be disabled.
4. Is the model loading from the same cache on every run? `ls -la ~/.cache/huggingface/hub/models--<...>/snapshots/`.

If all four are clean and the run is still nondeterministic, you've found a real D6 failure mode. Log it as a SETBACK and stop. This is a finding worth investigating, not bypassing.

### Lambda capacity vanishes mid-run

If a node disappears, the experiment is invalidated and must restart from the phase boundary. Lambda doesn't restart instances on hardware failure. You'll need to teardown and re-poll. The experiment log makes this less painful — you'll know exactly which phase to resume from.

### You're being charged after teardown

`python3 scripts/lambda/lambda_cli.py list` says nothing is running but the dashboard says you owe money. Check both:
- The web UI at `https://cloud.lambda.ai/instances`. Force-terminate from there.
- Any other clouds you might have running (vast.ai, Hyperbolic). The CLAUDE.md mentions both.

### You're stuck

Don't push through. Update the experiment log with what's stuck, what you've tried, and what hypothesis you have. Then ask. The log is the artifact that lets a second engineer help you in 5 minutes instead of 50.

---

## Appendix A: Codebase File Map

Files mentioned in this plan and what they do.

```
docs/plans/
├── d6-multinode-distributed-determinism.md   # the original D6 plan (read this first)
├── d6-lambda-staged-rollout.md               # this file
└── ...

cmd/
├── runner/
│   ├── main.py            # CLI entry point. Reads manifest+lockfile, dispatches.
│   ├── vllm_runner.py     # vLLM backend. Sets NCCL env vars (lines 15–36).
│   └── dispatcher.py      # picks backend by --mode
├── server/main.py         # vLLM API server. Not used in D6.
├── builder/main.py        # builds lockfiles from manifests
├── resolver/              # resolves HF artifacts to digests
├── verifier/              # canonical observable comparison
└── coordinator/           # multi-replica orchestration

pkg/manifest/model.py      # Pydantic models for v1 manifest
schemas/
├── manifest.v1.schema.json
└── lockfile.v1.schema.json

manifests/
├── dbrx-pp4-multinode.manifest.json           # Phase 3 (MoE; stub — resolver fills in HF shards)
├── dbrx-tp4-multinode.manifest.json           # Phase 3 (MoE; stub — resolver fills in HF shards)
├── mistral-large2-pp4-multinode.manifest.json # Phase 3 (dense)
├── mistral-large2-tp4-multinode.manifest.json # Phase 3 (dense)
├── mistral-large2-tp4.manifest.json           # original D4 manifest (template)
└── ...

lockfiles/                 # may not exist; generate per Task 3.5

scripts/
├── ci/
│   ├── d6_multinode_determinism.sh   # the test harness you'll run in Phase 3
│   └── ...
├── lambda/
│   └── lambda_cli.py      # the helper you'll write in Task 0.5
└── d6/                    # phase scripts you'll create as you go

experiments/
├── MULTI_GPU_DETERMINISM_REPORT.md   # D4 report (read for format reference)
├── d6-lambda-rollout-log.md          # YOU CREATE in Task 0.2
└── D6_MULTINODE_DETERMINISM_REPORT.md # YOU CREATE in Task 3.10

tests/determinism/
└── test_d4_batch_order_invariance.py # canonical determinism test pattern

flake.nix                  # the Nix recipe that built the container
```

---

## Appendix B: NCCL Environment Variables Reference

These are set automatically by `cmd/runner/vllm_runner.py:_set_deterministic_env` when `pp_size > 1` or `VLLM_MULTI_NODE` is set. You can also set them by hand if you're running a manual `python3` script outside the runner.

| Variable | Value | Purpose |
|----------|-------|---------|
| `NCCL_ALGO` | `Ring` | Pin to Ring algorithm. Prevents auto-tuning between Ring/Tree/CollNet, which changes reduction order. |
| `NCCL_PROTO` | `Simple` | Pin to Simple protocol. LL and LL128 introduce ordering variation. |
| `NCCL_NET` | `Socket` | Force TCP. No InfiniBand/RDMA, deterministic packet ordering via TCP. |
| `NCCL_P2P_DISABLE` | `1` | No GPU Direct P2P. Cross-node GPUs can't do direct P2P anyway. |
| `NCCL_SHM_DISABLE` | `1` | No shared memory. Cross-node has no shared memory available. |
| `NCCL_BUFFSIZE` | `8388608` | Pin to 8 MiB. Prevents buffer-size auto-tuning. |
| `NCCL_SOCKET_IFNAME` | `eth0` (default — **verify per node**) | Pin to a specific NIC. Prevents picking a different one across runs. **Lambda VMs may use `enp1s0`, `ens3`, etc., not `eth0`.** Run `ip -br link` on each node before relying on the default. Override by setting `NCCL_SOCKET_IFNAME` in the env before `vllm_runner.py` runs (Task 2.5). |
| `NCCL_DEBUG` | `WARN` | Logging level. Use `INFO` for diagnostics, `WARN` for normal runs. |

The `vllm_runner.py` logic is:
```python
if tp_size > 1 or pp_size > 1:
    env["NCCL_ALGO"] = "Ring"
    env["NCCL_PROTO"] = "Simple"
    env["NCCL_DEBUG"] = "WARN"
if pp_size > 1 or os.getenv("VLLM_MULTI_NODE"):
    env["NCCL_SOCKET_IFNAME"] = os.getenv("NCCL_SOCKET_IFNAME", "eth0")
    env["NCCL_NET"] = "Socket"
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_SHM_DISABLE"] = "1"
    env["NCCL_BUFFSIZE"] = "8388608"
```

---

## Appendix C: Lambda API Cheat Sheet

| Action | Endpoint | Method | Body |
|--------|----------|--------|------|
| List instances | `/instances` | GET | — |
| List instance types | `/instance-types` | GET | — |
| List SSH keys | `/ssh-keys` | GET | — |
| Add SSH key | `/ssh-keys` | POST | `{"name": "...", "public_key": "..."}` |
| Launch instance | `/instance-operations/launch` | POST | `{"region_name": "...", "instance_type_name": "...", "ssh_key_names": ["..."], "name": "..."}` |
| Terminate instance | `/instance-operations/terminate` | POST | `{"instance_ids": ["..."]}` |

Auth: HTTP Basic with API key as username, empty password.
```bash
curl -u "$LAMBDALABS_API_KEY:" https://cloud.lambdalabs.com/api/v1/instances
```

---

## Appendix D: The Container Run Recipe (canonical form)

You will type this many times. Copy-paste from here. Don't wrap it in a shell function; the YAGNI rule applies.

```bash
docker run --rm \
  --gpus all \
  --network host \
  --shm-size 10.24g \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/d6/manifests:/workspace/manifests-overlay \
  --entrypoint /bin/bash \
  ghcr.io/derpyplops/deterministic-serving:multinode \
  -c '<your command>'
```

For a long-running container that you'll `docker exec` into multiple times, replace `--rm` with `-d --name ray-head` (or `ray-worker`).

For Ray head:
```bash
docker run -d --name ray-head \
  --gpus all --network host --shm-size 10.24g --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/d6/manifests:/workspace/manifests-overlay \
  --entrypoint /bin/bash \
  ghcr.io/derpyplops/deterministic-serving:multinode \
  -c "ray start --head --node-ip-address=$(hostname -I | awk '{print $1}') --port=6379 --num-gpus=1 --block"
```

For Ray worker:
```bash
docker run -d --name ray-worker \
  --gpus all --network host --shm-size 10.24g --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/d6/manifests:/workspace/manifests-overlay \
  --entrypoint /bin/bash \
  ghcr.io/derpyplops/deterministic-serving:multinode \
  -c "ray start --address=$HEAD_IP:6379 --node-ip-address=$(hostname -I | awk '{print $1}') --num-gpus=1 --block"
```

Stop it with `docker stop ray-head` or `ray-worker`. Remove with `docker rm`.

---

## Final Note for the Engineer

Three things to remember:

1. **The plan is the abstraction.** Don't build a meta-plan, a poll-loop framework, an instance manager, or a "for next time" library. Do the tasks. Commit. Move on.

2. **The experiment log is the deliverable** as much as the report. If you finish all the tasks but the log is sparse, you did not finish the work. Future-you should be able to reconstruct exactly what happened from the log alone.

3. **The anti-cheat checks in Phase 2 are not optional.** They're the difference between "I ran PP=2 inference" and "I proved PP=2 inference is actually distributed across two nodes." We want the proof.

Good luck. When in doubt, log first, debug second, deviate last.
