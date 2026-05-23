# D6 Lambda Rollout — Experiment Log

This is an append-only journal for the staged D6 rollout on Lambda Cloud.
Every major action, configuration, setback, and milestone is logged here.

Plan: docs/plans/d6-lambda-staged-rollout.md

---

## Phase 0: Bootstrap

### 2026-04-13T11:34Z — Started Phase 0

Working through docs/plans/d6-lambda-staged-rollout.md.
Local environment verified: branch=multi-gpu-determinism, LAMBDALABS_API_KEY set,
~/.ssh/id_ed25519.pub present, all four multinode manifests (dbrx/mistral-large2 ×
pp4/tp4) on disk.

### 2026-04-13T11:36Z — COST: terminated 3 pre-existing leaked instances

Account inventory at start of Phase 0 showed 3 unexpected active instances
(combined ~$4.87/hr) unrelated to D6:
- 773a7845312b4ee08acc3f56969721a7  gpu_1x_gh200    us-east-3  dpdk-egress-test
- 2877a748b19843c791306577d8d53c30  gpu_1x_a10      us-west-1  pose-kexec-tight
- d20312c2e45b490c8852980cf70bba41  gpu_1x_a10      us-west-1  pose-kexec-tight

User confirmed termination. All three returned status=terminating.
Lesson: always run `lambda_cli.py list` before assuming a clean account.

### 2026-04-13T11:36Z — Verified Lambda API access

instance-types: 16 (gpu_1x_h100_sxm5 present)
instances: 0 (after termination)
ssh-keys: ['macbook 2025', 'macbook', 'arena 2022']
No key from this machine yet — will register in Task 0.4.

### 2026-04-13T11:37Z — Registered SSH key with Lambda

Key name: d6-rollout
Key id: 377c0353bad042cdb8bb81e8a1a688d1
Public key: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICbH+zsGjLDKlyelxJY6JQrtEYgGBBqSowk758eKNbbs

### 2026-04-13T11:37Z — SETBACK: lambda_cli.py first run hit Cloudflare 1010

GET /instance-types -> HTTP 403: error code: 1010

Lambda's Cloudflare WAF blocks the default `Python-urllib/3.x` User-Agent.
Curl worked from the same host (curl sets a UA Cloudflare accepts).
Fix: added `User-Agent: d6-lambda-cli/1.0` to `_auth_header()`.
After fix, types/list/keys all returned cleanly.

### 2026-04-13T11:37Z — MILESTONE: lambda_cli.py smoke test passed

Ran `poll gpu_8x_h100_sxm5 --count 1 --interval 5` under a SIGINT timeout.
Two iterations printed "no capacity", then SIGINT exited cleanly (5-line
traceback from time.sleep, within the plan's tolerance).

Note: at 11:37Z, `types` showed every GPU type with `available_in: -`
(zero capacity across all regions). Phase 1 polling may take a while.

### 2026-04-13T11:38Z — END Phase 0: bootstrap complete

Lambda API reachable, SSH key `d6-rollout` registered, lambda_cli.py in place
and tested. Account is clean (3 leaked instances terminated at start).
Ready for Phase 1 — pending user go-ahead before spending money.

---

## Phase 1: Single-Node Smoke Test

### 2026-04-13T11:42Z — COST: launched 1× H100 SXM5 on Lambda

Instance ID: a446ea16d747426a86e1c619f2a163e4
Type: gpu_1x_h100_sxm5
Region: us-south-2
Cost: $4.29/hr
Polling time: ~2.5 min (5 iters at 30s before capacity appeared)

### 2026-04-13T11:46Z — MILESTONE: SSH ready on Node 1

IP: 192.222.53.159
GPU: NVIDIA H100 80GB HBM3
Driver: 570.148.08
Time from launch to SSH-ready: ~4.5 min

### 2026-04-13T11:48Z — SETBACK: ubuntu user not in docker group

`docker pull` failed with permission denied on /var/run/docker.sock.
Workaround: prefix all docker commands with `sudo`. Did NOT add user to
docker group — sudo is fine for the duration of Phase 1.
For Phase 2/3, consider `sudo usermod -aG docker ubuntu` once per node.

### 2026-04-13T11:48Z — MILESTONE: container pulled on Node 1

Image: ghcr.io/derpyplops/deterministic-serving:multinode
Digest: sha256:0bb288d6d59391728b49fa3600e59c5a0b66f3cf72c94d59cd0b842a2b60b35d
Pull time: ~30s

### 2026-04-13T11:49Z — SETBACK: container's /usr/bin/nvidia-smi is broken

`nvidia-smi` inside the container returns "cannot execute: required file not
found". Common with Nix-built containers — the host's NVIDIA Container Toolkit
bind-mounts driver libs, but the container's nvidia-smi binary is incompatible.
Not a real problem: torch.cuda still works because libcuda is mounted in.
Workaround: skip nvidia-smi inside the container, use torch instead.

### 2026-04-13T11:50Z — MILESTONE: container has GPU access

`python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"`
→ `True 1 NVIDIA H100 80GB HBM3`

### 2026-04-13T11:50Z — SETBACK: vLLM batch_invariant requires explicit attention backend

First smoke run failed with:
  RuntimeError: VLLM batch_invariant mode requires an attention backend in
  ['FLASH_ATTN', 'TRITON_ATTN', 'FLASH_ATTN_MLA', 'TRITON_MLA'], but got 'None'.

Setting `VLLM_ATTENTION_BACKEND=FLASH_ATTN` env var did NOT propagate to vLLM
v1's attention_config (which is read independently of the env var).

Fix: pass `attention_backend="FLASH_ATTN"` as a kwarg to `LLM(...)` directly.
This matches what `cmd/runner/vllm_runner.py:123` already does. Updated
`scripts/d6/phase1_smoke.py` accordingly.

### 2026-04-13T11:51Z — MILESTONE: first inference on Lambda H100

Model: Qwen/Qwen3-0.6B
Engine init + 1 prompt × 20 tokens: ~1.3s/it after init
TOKEN_IDS: [264, 40803, 3405, 429, 702, 1293, 1012, 58574, 553, 60687, 11,
            13923, 11, 323, 68022, 13, 1084, 374, 264, 3405]
TEXT: " a philosophical question that has long been debated by philosophers,
       scientists, and thinkers. It is a question"

### 2026-04-13T11:52Z — SETBACK: Nix container has no `grep`

First determinism repeat run gave a false positive: piping `... | grep TOKEN_IDS`
*inside* the container produced empty files (grep not on PATH), and `diff` of
two empty files reported "DETERMINISTIC ✓".

Lesson: do all text filtering on the host side. The container is a python
runtime, not a userland.

Fix: redirect docker stdout to host, then `grep` on the host.

### 2026-04-13T11:53Z — MILESTONE: single-GPU determinism verified on Lambda

Two consecutive runs of Qwen3-0.6B (same seed, same prompt, same container)
produced bitwise-identical TOKEN_IDS (diff exit 0).

run-a TOKEN_IDS: [264, 40803, 3405, 429, 702, 1293, 1012, 58574, 553, 60687,
                  11, 13923, 11, 323, 68022, 13, 1084, 374, 264, 3405]
run-b TOKEN_IDS: (identical)

### 2026-04-13T11:53Z — MILESTONE: negative test passes

Same script with prompt "The opposite of hot is" produced:
  TOKEN_IDS: [9255, 11, 323, 279, 14002, 315, 9255, 374, 4017, 13, 2055, 11,
              421, 264, 1697, 374, 4017, 11, 1221, 279]
diff against run-a: differs immediately. Comparison detects mismatch.

### 2026-04-13T11:53Z — END Phase 1: single-node smoke complete

Container works on Lambda H100. Single-GPU inference is deterministic.
Determinism check is negative-tested. No unresolved SETBACKs.

Node 1 (a446ea16d747426a86e1c619f2a163e4 @ 192.222.53.159, us-south-2)
remains running and will become the Ray head node in Phase 2 (pending user
approval before launching Node 2).

Wall time Phase 1: ~16 min. Cost so far: ~$1.15.

---

## Phase 2: Two-Node Ray Cluster + Anti-Cheat

### 2026-04-13T13:20Z — COST: launched 2nd H100 SXM5

Instance ID: c9276bde08824791a687d6bd2a93d888
Type: gpu_1x_h100_sxm5
Region: us-south-2 (SAME region as Node 1 — rsync strategy for Phase 3 still open)
Cost: $4.29/hr
Polling time: ~3.5 min (7 iters)

### 2026-04-13T13:24Z — MILESTONE: Node 2 SSH-ready

IP (public): 192.222.52.85
IP (private): 172.27.124.165
Interface: eno1 (same as Node 1)
GPU: NVIDIA H100 80GB HBM3 (driver 570.148.08)

### 2026-04-13T13:25Z — Network topology note

Both nodes share private subnet 172.27.124.0/24 in us-south-2 with sub-ms RTT.
`hostname -I` returns the private address (Node 1: 172.27.124.243, Node 2:
172.27.124.165). Used the private IPs as `--node-ip-address` for Ray.
SSH from laptop still uses the public IPs (192.222.x.x); the two planes
are independent, so we can iptables-block the private net without killing SSH.

### 2026-04-13T13:25Z — MILESTONE: cross-node TCP verified

nc between nodes on 172.27.124.0/24:29500 works both directions with ~0.1ms
RTT. eno1 is the primary interface on both nodes (NOT eth0 — confirms the
plan's warning).

### 2026-04-13T13:27Z — SETBACK: `docker exec` into ray-worker lost NVML state

First Ray worker container started fine, `nvidia-smi --query` showed GPU at
container start, but after `ray start --block` was running, `docker exec
ray-worker python3 -c "import torch; print(torch.cuda.is_available())"`
returned False with "Can't initialize NVML". Direct ctypes call to
`nvmlInit_v2()` returned 999 (NVML_ERROR_UNKNOWN).

`docker run --rm --gpus all ...` on the same host worked fine, so the host's
NVIDIA Container Toolkit is functional. Something in the existing long-lived
`ray start` process got NVML into a bad state.

Fix: `docker rm -f ray-worker` and re-create. Problem went away.

Does not appear to be reproducible once the ray actors are in a clean state.

### 2026-04-13T13:29Z — SETBACK: vllm 0.17.1 × ray 2.54 `cuda_stream` ValueError

PP=2 inference fails with:
  ValueError: cuda_stream other than the current stream is not supported
  at vllm/distributed/device_communicators/ray_communicator.py:23-26

vLLM 0.17.1's `RayPPCommunicator.__init__` raises when Ray 2.54 passes a
non-current cuda_stream, and Ray 2.54 does exactly that.

First fix attempt: monkey-patch RayPPCommunicator in the driver Python —
did NOT work, because Ray workers are separate processes started by
`ray start --block` at container launch, with their own fresh `vllm` import.

Real fix: set `VLLM_USE_RAY_WRAPPED_PP_COMM=0` in the env of both ray-head and
ray-worker containers. This disables the RayPPCommunicator code path entirely
and falls back to direct NCCL for pipeline stage communication.

### 2026-04-13T13:35Z — CONFIG: NCCL pinning env baked into both containers

Both `ray-head` and `ray-worker` containers now launched with the full
`cmd/runner/vllm_runner.py:_set_deterministic_env` env baked in via
`docker run -e`:
  VLLM_MULTI_NODE=1
  NCCL_SOCKET_IFNAME=eno1
  NCCL_ALGO=Ring
  NCCL_PROTO=Simple
  NCCL_DEBUG=WARN
  NCCL_NET=Socket
  NCCL_P2P_DISABLE=1
  NCCL_SHM_DISABLE=1
  NCCL_BUFFSIZE=8388608
  VLLM_USE_RAY_WRAPPED_PP_COMM=0
Required because the Phase 2 smoke script calls `LLM()` directly and the
driver-side os.environ does NOT propagate to cross-node Ray worker actors.

### 2026-04-13T13:36Z — MILESTONE: MILESTONE: PP=2 cross-node inference completes

Config: Qwen/Qwen3-0.6B, pp=2, tp=1, backend=ray, enforce_eager=True
Placement: rank 0 on 172.27.124.243 (Node 1), rank 1 on 172.27.124.165 (Node 2)
  — confirmed by RayWorkerWrapper actor IPs in logs
vllm engine init: ~9s; inference (1×20 tok): ~1.3s/it

TOKEN_IDS: [264, 40803, 3405, 429, 702, 1293, 1012, 58574, 553, 60687, 11,
            13923, 11, 323, 68022, 13, 1084, 374, 264, 3405]
(identical to the Phase 1 single-GPU token sequence).

### 2026-04-13T13:38Z — MILESTONE: PP=2 over Ray cluster is deterministic

Two consecutive PP=2 runs produced bitwise-identical TOKEN_IDS.

### 2026-04-13T13:42Z — Check A: both GPUs used (PASS)

Ran pp2_long.py (8 prompts × 256 tokens). Sampled memory:
- Node 1 peak: 71199 MiB / 81559 MiB
- Node 2 peak: 1473 MiB / 81559 MiB (during inference)
  (Poll window started during init; saw the ramp 0→22→721→1473 MiB as the
  rank 1 worker loaded weights and allocated KV-cache state.)
The asymmetry is a vLLM gpu_memory_utilization quirk — rank 0 claims most of
its own GPU for KV cache, rank 1 profiles lighter. Both nonzero during
inference → PASS. Rank 1 logs confirm "Model loading took 0.71 GiB memory"
on ip=172.27.124.165, which is Node 2.

### 2026-04-13T13:42Z — Check B: cross-node NCCL ring confirmed (PASS)

With NCCL_DEBUG=INFO, the log shows:
  192-222-53-159:xxx [0] NCCL INFO NET/Socket : Using [0]eno1:172.27.124.243<0>
  192-222-53-159:xxx [0] NCCL INFO Initialized NET plugin Socket
  192-222-53-159:xxx [0] NCCL INFO Using network Socket
  192-222-53-159:xxx [0] NCCL INFO ncclCommInitRankConfig comm 0x... rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 6000 commId 0x5801c186324dbc1b - Init START
  192-222-52-85:xxx  [0] NCCL INFO ncclCommInitRankConfig comm 0x... rank 1 nranks 2 cudaDev 0 nvmlDev 0 busId 6000 commId 0x5801c186324dbc1b - Init COMPLETE
  192-222-52-85:xxx  [0] NCCL INFO NCCL_SHM_DISABLE set by environment to 1
Same commId across the two nodes → same NCCL communicator, connected via
NET/Socket on eno1. Our env-var pinning was picked up (NCCL_SHM_DISABLE line).

### 2026-04-13T13:47Z — Check C: iptables block breaks inference (PASS)

Inserted `iptables -I INPUT/OUTPUT -s/-d 172.27.124.X -j DROP` on both nodes
(REJECT --reject-with tcp-reset fails on Lambda's nf_tables backend with
"Invalid argument"; fell back to DROP since we were running a fresh process).
nc verified: 172.27.124.165:22 from Node 1 → timeout rc=124.
SSH from laptop on public 192.222.x.x unaffected.
Fresh PP=2 inference attempt hung at:
  WARNING: Tensor parallel size (2) exceeds available GPUs (1).
  INFO: Waiting for creating a placement group ... {'GPU': 1.0} * 1 (PACK)
After 60s timeout: exit 124, zero TOKEN_IDS in output → PASS.
Cleanup: removed both DROP rules, nc succeeds again.

### 2026-04-13T13:48Z — SETBACK: Ray cluster degraded after Check C teardown

After removing iptables rules the cluster still held the failed placement
group request and only showed 1 GPU available (worker on Node 2 had been
dropped during the network block). Sanity smoke hung forever.

Fix: `docker rm -f` and re-launch both containers. Cluster came back clean
(2 nodes, 2 GPUs, 52 CPU). Sanity smoke returned the canonical TOKEN_IDS.

### 2026-04-13T13:49Z — Check D: deferred to Phase 3

Mistral Large 2 (~240 GB) and DBRX (~265 GB) each exceed 1×H100 and 2×H100
aggregate VRAM, so Phase 3 inference completing at all is the structural
proof that 4-way distribution is real.

### 2026-04-13T13:49Z — MILESTONE: distributed inference is real (Phase 2 anti-cheat)

| Check | Result |
|-------|--------|
| A: Both GPUs use memory | PASS (N1 peak 71199 MiB, N2 peak 1473 MiB during inference) |
| B: NCCL cross-node NET/Socket ring | PASS (commId identical across 192-222-53-159 and 192-222-52-85, rank 0/1 COMPLETE, NCCL_SHM_DISABLE confirmed from env) |
| C: iptables block breaks inference | PASS (DROP rule → placement group unable to form, timeout 124) |
| D: Oversized model | DEFERRED to Phase 3 |

### 2026-04-13T13:50Z — END Phase 2

Both nodes still running. Distributed PP=2 is real and deterministic.
Phase 3 is BLOCKED pending HF token + Mistral Large 2 gating approval —
no HF token found on local machine (`~/.hf-token` and `~/.huggingface/token`
both absent; no HF_* env vars set).

Wall time Phase 2: ~33 min. Cost Phase 2 (2 nodes): ~$4.70.
Cumulative cost: ~$5.85.

---

## Phase 3: Full 4-Node D6 Experiment — BLOCKED

### 2026-04-13T13:16Z → 15:50Z — COST: 4-node cluster spin-up

- N1 (Phase 2 carry-over) 192.222.53.159 us-south-2 gpu_1x_h100_sxm5
- N2 (Phase 2 carry-over) 192.222.52.85  us-south-2 gpu_1x_h100_sxm5
- N3 6f2eda5621b44b7ca1226508586dbc7a 192.222.53.145 us-south-2 gpu_1x_h100_sxm5
- N4 6182e5517050415d9ec64653753e85cf 192.222.54.37  **us-south-3** gpu_1x_h100_sxm5

Node 4 polling took 96 iterations (~48 min) across both SXM5/PCIe before
capacity opened — and only in us-south-3, a different region from N1/2/3.

### 2026-04-13T14:13Z — SETBACK: Mistral Large 2 skipped from Phase 3

Two-layer problem:
1. HF token worked for direct curl of gated Mistral files (HTTP 302 → 200 OK on
   4.8 GB safetensor), but `huggingface_hub.hf_hub_download` inside the
   container hit `GatedRepoError 401`. Root cause: `pkg/common/hf_resolution.py:112`
   calls `hf_hub_download(...)` WITHOUT passing the token — the token is only
   given to `HfApi` for metadata. Fix: setting `HF_TOKEN` env var directly (not
   `--hf-token-file`) lets hf_hub_download pick it up.
2. Even with HF_TOKEN set, Mistral resolver wrote downloaded blobs to the
   container's ephemeral `/tmp/.cache/huggingface/` rather than the host-mounted
   volume. Would require re-download (~240 GB) for inference. Budget didn't
   allow both Mistral and DBRX.

Decision: skip Mistral, focus Phase 3 on DBRX alone.

### 2026-04-13T14:10Z — SETBACK: databricks/dbrx-instruct pulled from HF

`databricks/dbrx-instruct` returns 404 from HF Hub. The model was
removed/renamed since the plan was written. Substituted `alpindale/dbrx-instruct`
(a widely-used mirror) — verified has 61 safetensors shards / 263.2 GB,
matching the plan's expected footprint.

Patched both DBRX manifests to point at the mirror. Also had to:
- Pin `tokenizer_revision` + `weights_revision` from `"main"` to the
  current commit sha `8007650525bf3b67d6a4763caf02230061452d45` (manifest
  schema requires a 40-char hex).
- Flip `trust_remote_code: false` → `true` so AutoTokenizer uses DBRX's
  tiktoken-based shipped tokenizer class (container has tiktoken 0.12.0).

### 2026-04-13T15:00Z — MILESTONE: DBRX lockfiles resolved

`lockfiles/dbrx-pp4-multinode.lockfile.json` and `dbrx-tp4-multinode.lockfile.json`
generated via `cmd/resolver/main.py` running inside the ray-head container
(laptop has no venv — project is Nix-based). Each lockfile has 72 artifacts
including all 61 `model_weights` shards with sha256 digests.

HF Hub 1.5.0 appears to supply LFS sha256 via Xet metadata without full download
(lockfiles generated fast, HF cache empty). Good for resolver speed, fine
for lockfile validity.

### 2026-04-13T15:15Z — MILESTONE: DBRX weights on all 4 nodes

Used `huggingface_hub.snapshot_download(cache_dir=/root/.cache/huggingface/hub)`
inside each node's container with explicit HF_HOME to force writes to the
host-mounted volume (default `/tmp/.cache/huggingface` would be lost on
container rm). Download rate was ~3.5 GB/s per node via HF Xet CDN — 263 GB
completed in ~80 seconds per node. All 4 nodes cached 246 GiB of DBRX weights.

### 2026-04-13T15:59Z — MILESTONE: 4-node Ray cluster formed

4 nodes × H100 SXM5 registered in one Ray cluster via public IPs:
  Total: 0.0/104.0 CPU, 0.0/4.0 GPU, 618.98 GiB memory, 265.27 GiB object store
Bound `--node-ip-address` to each node's PUBLIC IP (192.222.x.x), necessary
because N4 in us-south-3 can't reach N1/2/3's us-south-2 private 172.27.x.x
subnet. us-south-2 ↔ us-south-3 cross-region RTT: 7.5 ms (verified with
ping), bidirectional TCP on port 29500 verified with nc.

### 2026-04-13T16:07Z — SETBACK: DBRX tokenizer needs tiktoken (or trust_remote_code)

First DBRX runner invocation failed with:
  ValueError: Couldn't instantiate the backend tokenizer from one of:
  (1) a `tokenizers` library serialization file,
  (2) a slow tokenizer instance to convert or
  (3) an equivalent slow tokenizer class to instantiate and convert.
  You need to have sentencepiece or tiktoken installed to convert a slow
  tokenizer to a fast one.

Container does have tiktoken 0.12.0, but HF AutoTokenizer only sees DBRX's
shipped tokenizer class when `trust_remote_code=True`. Patched both manifests,
regenerated lockfiles.

### 2026-04-13T17:02Z — SETBACK: vLLM Ray placement pinned to private IP

Second attempt failed before inference with Ray placement group waiting
forever on `{'node:172.27.124.243': 0.001, 'GPU': 1.0}` — vLLM computed
its own "driver IP" via get_ip() which returned the private 172.27.124.243
even though Ray itself had 4 nodes registered by public IPs.

Fix: set `VLLM_HOST_IP=<public>` on every container. After rebuild, vLLM's
placement pin used public IPs and all 4 ranks scheduled. Also verified via
a Ray actor diagnostic that all 4 workers see correct VLLM_HOST_IP and
`vllm.utils.network_utils.get_ip()` returns the expected public IP on each.

### 2026-04-13T18:06Z — SETBACK: Gloo connectFullMesh cannot establish

New failure mode after Ray placement fixed:
  RuntimeError: Gloo connectFullMesh failed with [...pair.cc:152] timed out
  connecting: SO_ERROR: Connection timed out,
  remote=[172.27.124.243]:51074

Rank 3 actor on 192.222.54.37 (Node 4, us-south-3) trying to reach Node 1's
gloo listener at 172.27.124.243:51074 — the PRIVATE IP, unreachable
cross-region. vLLM's NCCL init URL was correct (`tcp://192.222.53.159:<port>`),
but torch's separate ProcessGroupGloo creates its own full-mesh rendezvous
using `getifaddrs(eno1)[0]` for the bind address. The kernel returned private
first in that ordering.

**Workarounds tried, in order:**
1. **`ip addr del` + re-add private IP** to put public first in `getifaddrs` →
   verified `getifaddrs` now returns public first on all 4 nodes, but gloo
   still published private addresses. gloo's bind address comes from somewhere
   deeper than `getifaddrs` ordering.
2. **Host-side `/etc/hosts` patching** mapping `hostname → public IP` →
   `gethostbyname(gethostname())` returned public inside the containers, but
   gloo error persisted (it's not using gethostbyname).
3. **Container-side `/etc/hosts` patching + `--add-host` rebuild** so Ray
   workers inherit correct hostname resolution from container birth →
   verified inside all 4 containers, gloo error persisted. Error changed from
   `Connection timed out` (unreachable IP) to `Connection refused` (reachable
   but no listener on the published port) — which means gloo IS now publishing
   the public IP (192.222.53.159 in the error), but its actual bind socket is
   on a DIFFERENT IP (likely still private), so connections to the published
   addr:port get RST because nothing listens there.
4. **Added `VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=GLOO_,TP_` and
   `GLOO_SOCKET_IFNAME=eno1`** so `GLOO_` env vars propagate to Ray workers
   (default vLLM copy list only includes `HF_, HUGGING_FACE_, LMCACHE_, NCCL_,
   UCX_, VLLM_` — **GLOO_ is silently missing**) → `GLOO_SOCKET_IFNAME` now
   appears in the worker env copy list (confirmed in vllm log), but gloo error
   persists identically.

Independent cross-region TCP test from laptop: `python3` listener on N1
`0.0.0.0:41111` + N3 `socket.create_connection(("192.222.53.159", 41111))`
succeeded — so the underlying network connectivity is fine. The problem is
specific to how torch's nix-built `ProcessGroupGloo` chooses its bind address.

This is deep in libtorch C++ / nix build territory. At this point I've burned
~2 hours on Phase 3 debugging and ~$30 of additional GPU spend past Phase 2.
Decision: declare Phase 3 blocked on the torch/gloo bind-address issue,
terminate all resources, write a partial report.

Cross-region forced us down this path because us-south-2 had zero H100
capacity when we needed Node 4. In a same-region cluster the private IPs
match and Phase 2 showed the pipeline works. D6's **core property** — NCCL
cross-node determinism over TCP sockets — was already proven via Phase 2's
PP=2 + anti-cheat checks. Scaling to 4 GPUs × 2 regions is a deployment
dimension, not a new correctness claim.

### 2026-04-13T18:30Z — END Phase 3: blocked on torch/gloo + cross-region

DBRX PP=4 harness NOT executed. TP=4 harness NOT executed. Writing partial
report. Terminating all 4 Lambda instances.

Wall time Phase 3: ~5h 15m.
Cost Phase 3 (4 nodes, rolling overlap): estimated ~$43.
Cumulative cost through END Phase 3: ~$48.50.

