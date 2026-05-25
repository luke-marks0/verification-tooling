# D6: Multi-Node Distributed Determinism Testing

## Motivation

### What we've already proven

The deterministic serving stack has established, through a series of experiments (D1–D4), that LLM inference can be made bitwise-reproducible:

| Experiment | What it proved | GPU | Result |
|-----------|---------------|-----|--------|
| D1 — Single-GPU cross-server | Same model, same prompt, two independent GH200 servers → same tokens | 1× GH200 | 91/91 chunks match |
| D2 — Single-GPU MoE | MoE expert routing is deterministic (Qwen3-30B-A3B) | 1× GH200 | PASS |
| D3 — Batch/order invariance (single GPU) | Shuffling requests + changing batch size doesn't change per-request output (H100 only) | 1× H100 | PASS |
| D4 — Tensor parallel (single node) | Sharding across 4 GPUs on one machine is deterministic | 4× RTX 4090, 4× H100 | PASS |
| D4 — TP batch/order invariance | TP + shuffled order + different batch size → same output (H100 only) | 4× H100 | PASS (Mistral Large 2 + Llama 4 Scout) |

Full report: `experiments/MULTI_GPU_DETERMINISM_REPORT.md`

### What D6 will prove

**Can we get deterministic inference when GPUs are on separate physical machines, communicating over TCP?**

This is the hardest test. Single-node multi-GPU uses NVLink (600 GB/s, hardware-ordered). Multi-node uses NCCL over TCP sockets — packets traverse switches, NICs, kernel network stacks. We need to prove that pinning NCCL's algorithm (Ring) and protocol (Simple) is sufficient to make cross-node collectives deterministic.

Two parallelism strategies:

- **Pipeline Parallel (PP=4)**: Each node holds 1/4 of the model's layers. Point-to-point NCCL send/recv between adjacent pipeline stages. Lower communication volume.
- **Tensor Parallel (TP=4 over TCP)**: Each node holds 1/4 of every layer's weight matrices. All-reduce at every layer over TCP. High communication volume, but directly extends our single-node TP proof.

### Why this matters

Production LLM deployments serving models larger than a single GPU's memory (70B+, 123B+, 405B) use distributed inference. If we can prove determinism holds over the network, the deterministic serving stack covers real production topologies — not just lab setups.

---

## Current State

### Infrastructure already provisioned

| Resource | Details | Status |
|----------|---------|--------|
| **Node 0 (Head)** | vast.ai instance `34551109`, 1× H100 SXM 80GB, `ssh9.vast.ai:31108`, $1.72/hr | Running |
| **OCI container** | `ghcr.io/derpyplops/deterministic-serving:multinode` (6.6GB) | Pushed to GHCR |
| **OCI container (latest)** | `ghcr.io/derpyplops/deterministic-serving:latest` | Pushed to GHCR |

### Sanity checks completed on Node 0

- Nix dev shell: torch 2.10.0, vLLM 0.17.1, Ray 2.54.0
- GPU access: H100 80GB HBM3, driver 580.126.20, FlashAttention v3
- Inference: Qwen3-0.6B loaded, generated tokens, determinism verified (two identical runs → same output)
- All code changes present on the node (pulled from `multi-gpu-determinism` branch)

### Code changes already merged (on `multi-gpu-determinism` branch)

1. **`modules/inference/runner/vllm_runner.py`** — Multi-node NCCL env vars (`NCCL_NET=Socket`, `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_BUFFSIZE=8388608`) when `pp_size > 1` or `VLLM_MULTI_NODE` is set. Passes `distributed_executor_backend` to vLLM `LLM()` constructor.
2. **`pkg/manifest/model.py`** — Added `distributed_executor_backend: str | None = None` to `ServingEngine`.
3. **`schemas/manifest.v1.schema.json`** — Added `"distributed_executor_backend": {"enum": ["ray", "mp"], "type": "string"}` to serving_engine properties.
4. **`modules/inference/server/main.py`** — Added `--distributed-executor-backend` CLI flag passthrough.
5. **`manifests/qwen3-30b-moe-pp4-multinode.manifest.json`** — PP=4, Ray backend, H100 hardware profile, batch invariance enabled.
6. **`manifests/qwen3-30b-moe-tp4-multinode.manifest.json`** — TP=4, Ray backend, H100, batch invariance enabled.
7. **`scripts/deploy/vast/grab_cluster.sh`** — Interactive script to provision 4 vast.ai nodes.
8. **`scripts/deploy/vast/setup_cluster.sh`** — Sets up Ray cluster across 4 nodes.
9. **`scripts/deploy/vast/teardown_cluster.sh`** — Destroys vast.ai instances.
10. **`scripts/ci/d6_multinode_determinism.sh`** — Full test harness (same-config, batch+order invariance, TP-over-TCP stretch).

---

## Execution Plan

### Step 1: Rent 3 Worker Nodes

Search for H100 SXM instances, preferably in the US (same region as Node 0) for low latency:

```bash
vastai search offers \
  'gpu_name=H100_SXM num_gpus=1 cuda_vers>=12.0 reliability>0.90 inet_down>300 disk_space>80' \
  -o 'dph'
```

Create 3 instances using the GHCR image (workers don't need to build anything):

```bash
vastai create instance <offer_id> \
  --image ghcr.io/derpyplops/deterministic-serving:multinode \
  --disk 100
```

Repeat for 3 different offers. Record the contract IDs and SSH info.

**Important**: Workers boot from the OCI image directly. No Nix install needed.

**Cost**: ~$1.50-2.00/hr per node. Total cluster: ~$6-8/hr.

### Step 2: Verify Network Connectivity

Each node needs direct IP connectivity to all others. From each node:

```bash
# Get each node's internal IP
hostname -I | awk '{print $1}'

# From each node, verify TCP connectivity to all others on NCCL port range
for ip in $NODE0_IP $NODE1_IP $NODE2_IP $NODE3_IP; do
    nc -zv $ip 29500 && echo "$ip: OK" || echo "$ip: FAIL"
done
```

**If connectivity fails**: vast.ai instances on different hosts may not have direct IP access. Options:
- Use `direct_port_count` filter when searching offers
- Rent all 4 from the same host (look for `host_id` in offer details)
- Fall back to SSH tunneling (last resort, adds latency)

### Step 3: Form Ray Cluster

On **Node 0** (head):
```bash
pip install ray[default]  # If not in container
INTERNAL_IP=$(hostname -I | awk '{print $1}')
ray start --head --port=6379 --dashboard-host=0.0.0.0 --node-ip-address="$INTERNAL_IP"
```

On each **worker node** (Nodes 1-3):
```bash
pip install ray[default]  # If not in container
INTERNAL_IP=$(hostname -I | awk '{print $1}')
ray start --address=$HEAD_IP:6379 --node-ip-address="$INTERNAL_IP"
```

Verify on head:
```bash
ray status  # Should show 4 nodes, 4 GPUs
```

### Step 4: Download Model on All Nodes

Each node needs a local copy of the model weights:

```bash
# On all 4 nodes (can run in parallel):
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-30B-A3B')
print('Done')
"
```

Qwen3-30B-A3B is ~60GB, ungated (no HF token needed). ~10 min per node at 1 Gbps.

### Step 5: Run the Experiments

On the **head node**, set the environment and run:

```bash
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NET=Socket
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_BUFFSIZE=8388608
export NCCL_SOCKET_IFNAME=eth0  # or whatever the container's interface is
export NCCL_DEBUG=WARN
export VLLM_MULTI_NODE=1
export VLLM_BATCH_INVARIANT=1
export RAY_ADDRESS=auto
export LD_LIBRARY_PATH=/tmp/cuda-driver  # If using nix dev shell, not OCI

cd /root/deterministic_serving_stack
./scripts/ci/d6_multinode_determinism.sh
```

Or run tests individually:

#### Test 1: PP=4 Same-Config Determinism
Run the same 100 requests twice with identical config. Compare by request ID.
```bash
# Run A
python3 modules/inference/runner/main.py --manifest manifests/qwen3-30b-moe-pp4-multinode.manifest.json \
  --lockfile lockfiles/qwen3-30b-moe.lockfile.json --out-dir /tmp/d6/pp4-run-a --mode vllm --replica-id replica-0

# Run A' (identical)
python3 modules/inference/runner/main.py --manifest manifests/qwen3-30b-moe-pp4-multinode.manifest.json \
  --lockfile lockfiles/qwen3-30b-moe.lockfile.json --out-dir /tmp/d6/pp4-run-a-prime --mode vllm --replica-id replica-0

# Compare
python3 -c "
import json, glob
def load(path):
    f = glob.glob(f'{path}/replica-*/observables.json')[0]
    return {r['id']: r for r in json.load(open(f))['request_outputs']}
a, b = load('/tmp/d6/pp4-run-a'), load('/tmp/d6/pp4-run-a-prime')
match = sum(1 for k in a if a[k]['tokens'] == b[k]['tokens'])
print(f'{match}/{len(a)} match')
"
```
**Expected**: 100/100 match.

#### Test 2: PP=4 Batch + Order Invariance
Run A: 100 requests in original order, `max_num_seqs=64`.
Run B: same 100 requests shuffled (seed 12345), `max_num_seqs=16`.
Compare outputs matched by request ID.

**Expected**: 100/100 match (H100 with `VLLM_BATCH_INVARIANT=1`).

#### Test 3 (Stretch): TP=4-over-TCP Same-Config
Use `manifests/qwen3-30b-moe-tp4-multinode.manifest.json` instead. This is TP=4 across 4 nodes over TCP — all-reduce at every layer. Much slower than PP, but directly extends the single-node TP proof.

**Expected**: 100/100 match (if NCCL Ring+Simple is deterministic over TCP).

### Step 6: Collect Results and Teardown

```bash
# Copy results locally
scp -r -P 31108 root@ssh9.vast.ai:/tmp/d6/ experiments/multinode_determinism/

# Destroy all instances
vastai destroy instance <id1>
vastai destroy instance <id2>
vastai destroy instance <id3>
vastai destroy instance <id4>
```

---

## Architecture Diagram

```
Node 0 (Head)           Node 1 (Worker)        Node 2 (Worker)        Node 3 (Worker)
┌─────────────────┐     ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│ 1× H100 SXM 80GB│◄───►│ 1× H100 SXM    │◄───►│ 1× H100 SXM    │◄───►│ 1× H100 SXM    │
│ Ray Head         │     │ Ray Worker     │     │ Ray Worker     │     │ Ray Worker     │
│ vLLM engine      │     │ vLLM worker    │     │ vLLM worker    │     │ vLLM worker    │
│ Runner script    │     │                │     │                │     │                │
└─────────────────┘     └────────────────┘     └────────────────┘     └────────────────┘
       │                       │                      │                      │
       └───────────── NCCL over TCP (Ring + Simple) ──┘──────────────────────┘
```

**PP=4 data flow**: Input → Node 0 (layers 0-14) → Node 1 (layers 15-29) → Node 2 (layers 30-44) → Node 3 (layers 45-59) → Output. Point-to-point NCCL send/recv between adjacent stages.

**TP=4 data flow**: Each node has 1/4 of every layer's weights. All-reduce across all 4 nodes at every layer boundary. High bandwidth requirement.

---

## NCCL Determinism Settings (Critical)

These environment variables pin NCCL's behavior to prevent non-deterministic algorithm/protocol selection:

| Variable | Value | Why |
|----------|-------|-----|
| `NCCL_ALGO=Ring` | Fixed collective algorithm | Prevents auto-tuning between Ring/Tree/CollNet, which can change reduction order |
| `NCCL_PROTO=Simple` | Fixed protocol | LL and LL128 protocols introduce ordering variation |
| `NCCL_NET=Socket` | Force TCP sockets | No InfiniBand/RDMA — deterministic packet ordering via TCP |
| `NCCL_P2P_DISABLE=1` | No GPU Direct P2P | Cross-node: GPUs can't do direct P2P anyway |
| `NCCL_SHM_DISABLE=1` | No shared memory | Cross-node: no shared memory available |
| `NCCL_BUFFSIZE=8388608` | Fixed 8MB buffer | Prevents buffer size auto-tuning |
| `NCCL_SOCKET_IFNAME=eth0` | Pin network interface | Prevents NCCL from picking different NICs across runs |
| `NCCL_DEBUG=WARN` | Logging | Helps diagnose issues without flooding output |

These are set automatically by `vllm_runner.py` when `pp_size > 1` or `VLLM_MULTI_NODE=1`.

---

## Model Details

**Qwen3-30B-A3B** (Mixture of Experts)
- 30B total parameters, 3B active per token
- 128 experts per MoE layer
- 16 weight shards (safetensors), ~60GB total
- HF revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
- Ungated (no HF token required)
- Source: `hf://Qwen/Qwen3-30B-A3B`

We chose an MoE model because expert routing adds another potential source of non-determinism — if routing decisions diverge across runs, everything downstream diverges too. Proving MoE determinism over the network is strictly harder than proving dense model determinism.

---

## The Determinism Recipe

```
# Parallelism
pipeline_parallel_size = 4        # or tensor_parallel_size = 4
distributed_executor_backend = ray
disable_custom_all_reduce = true

# vLLM
enforce_eager = true              # No CUDA graphs
VLLM_BATCH_INVARIANT = 1          # H100 only: batch invariance
attention_backend = FLASH_ATTN    # FlashAttention v3 on Hopper

# PyTorch
seed = 42
torch_deterministic = true
CUBLAS_WORKSPACE_CONFIG = :4096:8
CUDA_LAUNCH_BLOCKING = 1
PYTHONHASHSEED = 0

# NCCL (multi-node)
NCCL_ALGO = Ring
NCCL_PROTO = Simple
NCCL_NET = Socket
NCCL_P2P_DISABLE = 1
NCCL_SHM_DISABLE = 1
NCCL_BUFFSIZE = 8388608
NCCL_SOCKET_IFNAME = eth0
```

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Cross-node networking fails** | Medium | Blocking | Verify connectivity before forming Ray cluster. Fall back to same-host multi-GPU if needed. |
| **NCCL over TCP is non-deterministic** | Low | Experiment fails | TCP guarantees ordered delivery. Ring+Simple should be deterministic. If not, try `NCCL_MAX_NCHANNELS=1` to force single-channel communication. |
| **Ray cluster instability** | Low | Lost runs | Use high-reliability instances (>0.90). Run in tmux. Save intermediate results. |
| **Ghost instance billing** | Medium | Cost overrun | Set a timer. Check `vastai show instances` before sleeping. Destroy immediately after experiment. |
| **Model download takes too long** | Low | Delay | Download on all 4 nodes in parallel. HF cache preserves partial downloads. |
| **OCI image incompatible with worker hosts** | Low | Workers fail to boot | All H100 SXM instances should be compatible. Check CUDA driver version ≥ 12.0. |

---

## Cost Estimate

| Item | Duration | Rate | Cost |
|------|----------|------|------|
| Node 0 (already running, OCI build done) | ~1hr remaining | $1.72/hr | ~$1.72 |
| Nodes 1-3 (workers) | ~1hr each | ~$1.60/hr each | ~$4.80 |
| **Total remaining** | | | **~$6.50** |
| Total including OCI build time (~2hr) | | | **~$10** |

---

## Files Reference

| File | Purpose |
|------|---------|
| `modules/inference/runner/vllm_runner.py` | Offline vLLM inference backend with multi-node NCCL support |
| `modules/inference/runner/main.py` | Runner entry point |
| `modules/inference/server/main.py` | vLLM server CLI with `--distributed-executor-backend` |
| `pkg/manifest/model.py` | Pydantic manifest models (ServingEngine has `distributed_executor_backend`) |
| `schemas/manifest.v1.schema.json` | JSON Schema for manifest validation |
| `manifests/qwen3-30b-moe-pp4-multinode.manifest.json` | PP=4 multi-node manifest |
| `manifests/qwen3-30b-moe-tp4-multinode.manifest.json` | TP=4 multi-node manifest |
| `scripts/deploy/vast/grab_cluster.sh` | Provision 4 vast.ai nodes |
| `scripts/deploy/vast/setup_cluster.sh` | Form Ray cluster across nodes |
| `scripts/deploy/vast/teardown_cluster.sh` | Destroy vast.ai instances |
| `scripts/ci/d6_multinode_determinism.sh` | Full D6 test harness |
| `experiments/MULTI_GPU_DETERMINISM_REPORT.md` | D4 results report (prior experiment) |

---

## Nix Dev Shell Gotcha

When running inference on a vast.ai node using `nix develop` (not the OCI container), the host CUDA driver isn't on the Nix library path. Fix:

```bash
# Create isolated driver symlinks (avoids glibc conflicts)
mkdir -p /tmp/cuda-driver
for lib in libcuda libnvidia-ml libnvidia-ptxjitcompiler libnvidia-nvvm libnvidia-gpucomp; do
    for f in /usr/lib/x86_64-linux-gnu/${lib}.so*; do
        [ -f "$f" ] && ln -sf "$f" /tmp/cuda-driver/
    done
done
export LD_LIBRARY_PATH=/tmp/cuda-driver

# Do NOT add /usr/lib/x86_64-linux-gnu directly — causes stack smashing from glibc conflicts
```

When using the OCI container (workers), the driver is bind-mounted automatically via `--gpus all`.
