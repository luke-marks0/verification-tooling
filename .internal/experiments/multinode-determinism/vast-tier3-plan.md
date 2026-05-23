# D6 Tier 3 Shuffle Extension on Vast.ai

**Scope:** run the three Tier 3 experiments that didn't execute on Lambda due to capacity: DBRX A==B-shuffled at 1M tokens, Mistral Large 2 A==A' at 1M tokens, Mistral Large 2 A==B-shuffled at 1M tokens. Move the whole thing to vast.ai because Lambda H100 capacity has been zero for ~24h and burning idle nodes waiting for it is the worst possible spend pattern.

**Prerequisite result this extends:** `experiments/D6_MULTINODE_DETERMINISM_REPORT.md`. Tier 1, Tier 2 (both models, both A==A' and A==B-shuffled), and Tier 3 DBRX A==A' (549,927 tokens) are already proven on Lambda. Nothing in this plan is load-bearing for the existing D6 claim — it's the scale-up extension that was descoped when Lambda capacity dried up.

**Companion docs:**
- `docs/plans/d6-lambda-staged-rollout.md` — staged execution pattern, container canon
- `docs/plans/d6-determinism-tiers.md` — 3-tier workload spec and comparison semantics
- `experiments/D6_MULTINODE_DETERMINISM_REPORT.md` — prior results and all 9 Lambda-surfaced bugs

---

## Why vast.ai and why a different image

Lambda gives you a real Linux VM per instance; `--network host` in Docker shares the VM's network namespace, and NCCL/gloo see the VM's eno1 addresses directly. **Vast.ai is fundamentally different**: each instance is a Docker container on a marketplace host you don't control, fronted by a vast-managed proxy that maps container ports to the host's public IP on some high port (e.g. `32768`), not to port 22 directly. For SSH to work end-to-end, the container **must run an sshd of its own** — the Nix-built `:multinode` image we used on Lambda does not.

The `ghcr.io/derpyplops/deterministic-serving:vast-test` tag is the variant with sshd baked into the Nix closure. Same vLLM 0.17.1 / Ray 2.54 / PyTorch 2.10 stack, plus:

- `openssh`, `/etc/nsswitch.conf`, unlocked `/etc/shadow`, an `sshd` user, and `/var/{empty,log,run}` baked in (Nix-only rootfs is normally missing these bits).
- An entrypoint at `/nix/store/<hash>-entrypoint/bin/entrypoint` which:
  1. Appends a pubkey to `/root/.ssh/authorized_keys` from env (preferring `PUBKEY_B64`, falling back to `SSH_PUBLIC_KEY` or `PUBLIC_KEY`).
  2. If `SKIP_SERVER` is set, runs `sshd -D -e` in foreground only (no vLLM server) — use for debugging / smoke tests.
  3. Otherwise backgrounds sshd and `exec`s `python3 cmd/server/main.py "$@"`. `main.py` requires `--manifest <path>` or it exits and vast restarts the container, so for this experiment we pass `SKIP_SERVER=1` and drive everything via `docker exec` like we did on Lambda.

**Critical:** the entrypoint's full store path (`/nix/store/<hash>-entrypoint/bin/entrypoint`) is **not stable** — it changes on every Nix rebuild of the image. You must discover it from the published image manifest **at launch time**; hard-coding yesterday's hash is a guaranteed break.

```bash
TOK=$(gh auth token)
DIGEST=$(curl -sL -H "Authorization: Bearer $(echo -n $TOK | base64)" \
  -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
  "https://ghcr.io/v2/derpyplops/deterministic-serving/manifests/vast-test" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['config']['digest'])")
ENTRY=$(curl -sL -H "Authorization: Bearer $(echo -n $TOK | base64)" \
  "https://ghcr.io/v2/derpyplops/deterministic-serving/blobs/$DIGEST" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['config']['Cmd'][0])")
# ENTRY now holds the correct /nix/store/<hash>-entrypoint/bin/entrypoint
```

**Unknowns we must validate before spending big:**

1. Does `vast-test` contain the same Python environment (vLLM 0.17.1, Ray 2.54.0, torch 2.10, FlashAttention v3, hf_hub 1.5.0, tiktoken 0.12.0) that we relied on on Lambda? If it's a fresh Nix build, the sha256s will differ. Lockfile `manifest_digest` checks still hold because they pin manifest→sha not container→sha, but the runner's Python-level call sites must match (they should, same flake). **The runner's Nix store path — `/nix/store/chlbq1wd9fbsia3lrsxhj2qcw9z30823-deterministic-serving-stack-0.1.0` from the Lambda session — will almost certainly differ on vast-test**; discover the real path before copy-pasting any `cd /nix/store/...` commands from the Lambda runbook.
2. Cross-host NCCL/gloo — **the main risk**. See "Networking reality check" below.

---

## Networking reality check (the most likely showstopper)

On Lambda all four nodes were real VMs with a dual-stack `eno1`, and the original session's gloo-bind fix (`scripts/d6/sitecustomize.py`) worked by calling `ProcessGroupGloo.create_device(hostname=$VLLM_HOST_IP)`. That fix does not magically solve vast's NAT.

**Vast container network options, in order of preference for this experiment:**

1. **`--network host` on a direct-host vast offer.** A subset of vast offers advertise `direct_port_count` and support host-mode networking. The container binds directly to the physical host's network stack, same as Lambda, and our existing pinning + sitecustomize patch should transfer unchanged. **Filter for this when searching offers.** `vastai search offers 'direct_port_count>=20'` or similar.
2. **Bridged networking with per-port exposes.** Vast lets you specify `-p <host_port>:<container_port>` at instance creation, but the port list must be static and known in advance. NCCL and gloo allocate ephemeral ports on every init — you can't enumerate them. This is the mode that defeated the original plan ("D6 over vast.ai was blocked because Docker NAT breaks Ray/NCCL across hosts").
3. **Wireguard or ZeroTier overlay.** Third path: install a mesh VPN on every instance after launch, have every node join a private overlay subnet, and pretend it's a private LAN. Adds ~30 min of setup per node and one more failure mode, but bypasses the NAT question entirely. Used as fallback if host-mode offers aren't available.

**Go/no-go on option 1:** the cluster will form at all iff vast has ≥4 offers with host-mode networking for H100 SXM5 or H100 PCIe at the time of polling. If it doesn't, this plan stops and falls back to option 3 (overlay) or aborts.

**Go/no-go on option 3:** provisioning WireGuard across 4 containers, distributing keys, picking a /24, and having every container join the same interface before the Ray cluster forms is doable but adds an entire new axis of things that can break. It's the "we committed and have to see this through" fallback, not the happy path.

---

## Prerequisites

### Account / credentials

- **Vast.ai account + API key.** Set `VAST_API_KEY` in the shell. (`CLAUDE.md` does not currently mention vast credentials, so the user must add the key locally.)
- **`vastai` CLI installed via uv.** `uv tool install vastai` (preferred) or `uv pip install vastai`.
- **SSH keypair registered with vast.** `vastai create ssh-key --ssh-key "$(cat ~/.ssh/id_ed25519.pub)"` (one-time).
- **HuggingFace token** with access to `mistralai/Mistral-Large-Instruct-2407` (gated) and `alpindale/dbrx-instruct` (public). We already have the token working — use the same one from the Lambda session.

### Repo state

- `multi-gpu-determinism` branch checked out with all prior commits including:
  - `scripts/d6/sitecustomize.py` (the gloo bind patch)
  - `manifests/dbrx-tp4-large.manifest.json`, `manifests/dbrx-tp4-large-shuffled.manifest.json`
  - `manifests/mistral-large2-tp4-large.manifest.json`, `manifests/mistral-large2-tp4-large-shuffled.manifest.json`
  - `lockfiles/dbrx-tp4-large.lockfile.json` (the existing one pins the DBRX manifest — will need regen against the container's resolver if the vast-test container is a different Nix build)
- Baseline observables on disk for comparison:
  - `experiments/multinode_determinism/20260414/tier3-large/dbrx/dbrx-large-a/observables/tokens.json` (549,927-token Lambda DBRX A)

### Tooling

- `scripts/d6/compare_observables.py` (exists) — token-exact per-request divergence reporter, no changes needed.
- `scripts/d6/prompts.py` / `scripts/d6/generate_tier_manifests.py` — shouldn't need to touch, but keep them handy if any manifest needs a re-gen.

---

## Experiments to run

Labeled `T3V-*` for "Tier 3 on Vast" so the result paths don't collide with the Lambda baselines already committed.

| Label | Manifest | Purpose | Compared against |
|---|---|---|---|
| **T3V-dbrx-a** | `dbrx-tp4-large.manifest.json` | Optional cross-platform sanity: does Lambda DBRX-A match vast DBRX-A? If yes: bonus cross-platform determinism. If no: two clusters give different bits — new finding, stop and investigate. | Lambda `dbrx-large-a` (already committed) |
| **T3V-dbrx-b** | `dbrx-tp4-large-shuffled.manifest.json` | **The extension itself** — DBRX batch-order invariance at 549K+ tokens. | Lambda `dbrx-large-a` (same manifest, shuffled prompts, compared per-id) |
| **T3V-mistral-a** | `mistral-large2-tp4-large.manifest.json` | Mistral Large 2 at the 1M-token tier — no baseline exists. | T3V-mistral-aprime (also new) |
| **T3V-mistral-aprime** | same as above | Same-config repeat | T3V-mistral-a |
| **T3V-mistral-b** | `mistral-large2-tp4-large-shuffled.manifest.json` | Mistral Large 2 batch-order invariance at 1M tokens. | T3V-mistral-a (per-id) |

Run order:
1. `T3V-dbrx-a` (cross-platform check, cheap abort signal if something is off)
2. `T3V-dbrx-b`
3. `T3V-mistral-a`
4. `T3V-mistral-aprime`
5. `T3V-mistral-b`

Five runs × ~80 min each ≈ **6.5 hours of pure inference** if throughput matches Lambda's. Vast hosts vary in CPU, NVLink generation, and driver version — expect ±30% variance from the Lambda 78-min-per-run baseline.

### Optional: drop T3V-dbrx-a to save one run

If cost is tight, skip T3V-dbrx-a. The reasoning: we already have high confidence in intra-cluster DBRX determinism from Lambda A==A'. T3V-dbrx-b compared against Lambda DBRX-A still tests batch-order invariance — it's a cross-platform A==B check in one shot. The only thing that T3V-dbrx-a would add is "vast platform does NOT introduce per-run variance vs Lambda platform," which is a bonus rather than core.

Running T3V-dbrx-a is recommended because if T3V-dbrx-b fails and you haven't run T3V-dbrx-a, you can't tell whether the failure is from the shuffle (real D6 finding!) or from vast-vs-lambda platform variance (infrastructure noise). **Run A first** unless budget is screaming.

---

## Instance acquisition

Constraints for a valid D6 cluster:

- **GPU**: 4 × H100 80GB (SXM5 preferred, PCIe acceptable). The workload is TP=4 over TCP — we're not bottlenecked on NVLink, so PCIe is fine.
- **Host-mode networking**: must be available (see Networking reality check above). Search filter: `verified=true num_gpus=1 gpu_name=H100 direct_port_count>=20`.
- **Disk**: ≥800 GB free after image pull. DBRX (246 GB) + Mistral (456 GB with both consolidated and sharded variants) + HF metadata + room = ~750 GB of pure workload data. Vast charges per GB-hour for disk, so don't oversize.
- **Reliability**: only `verified=true` hosts. Vast's DLP rating >0.9 preferred.
- **Geography**: no preference — the gloo fix makes cross-region work, and vast's host-mode networking sidesteps the dual-stack issue entirely (each container has one real public interface).
- **Price cap**: set an upper bound based on the current market. Typical H100 SXM5 on vast: **$1.30–$2.50/hr per GPU**. Four nodes × 80min × 5 runs × $2.00/hr ≈ **~$55**. Plus overhead (image pulls, model downloads, waiting): budget **~$75–100** total.

```bash
# Search for candidates (run this before creating anything)
vastai search offers \
  'verified=true rentable=true num_gpus=1 gpu_name=H100 \
   gpu_ram>=80 disk_space>=800 cpu_ram>=128 \
   direct_port_count>=20 inet_down>=1000 inet_up>=1000 \
   dlperf>25' \
  -o 'dph+'  # sort by dollars-per-hour ascending
```

The `direct_port_count` filter is the important one for our networking model — it filters to hosts that advertise enough directly-routable ports to support `--network host` or a wide `-p` mapping.

Pick 4 offers, prefer similar hardware and reasonable prices. Record their `id` values.

**Must use `--args` mode, NOT `--ssh` mode.** Vast's `--ssh` flag injects a `/.launch` wrapper that assumes a Ubuntu-style rootfs (`apt-get`, `grep`, `/usr/sbin/sshd`) and crash-loops on the Nix-only image. `--args` mode runs the container's real entrypoint unmodified and lets vast map the ports you request via `-p` in the env bundle.

```bash
PUBKEY_B64=$(cat ~/.ssh/id_ed25519.pub | base64 -w0)
# $ENTRY was discovered from the image manifest in "Why vast.ai..." above.

for OFFER in $OFFER_1 $OFFER_2 $OFFER_3 $OFFER_4; do
  vastai create instance $OFFER \
    --image ghcr.io/derpyplops/deterministic-serving:vast-test \
    --disk 800 \
    --entrypoint $ENTRY \
    --env "-p 22:22 -p 6379:6379 -p 29500:29500 -p 29501:29501 -p 29502:29502 -p 29503:29503 -e PUBKEY_B64=$PUBKEY_B64 -e SKIP_SERVER=1" \
    --args
done
```

Notes on the `create instance` flags:

- `--args` (not `--ssh`) — gives vast a stub `CMD` and lets our `--entrypoint` run the real sshd.
- `--image` is the vast-test tag. Vast pulls this from ghcr — public tag, no auth needed.
- `--disk 800` budgets 800 GB for DBRX (246 GB) + Mistral Large 2 (456 GB with both consolidated and sharded formats) + HF metadata + headroom.
- `--entrypoint $ENTRY` forces vast to use the real Nix-store entrypoint we discovered from the image manifest. Must be recomputed per-day because it changes every rebuild.
- `--env` is a vast-style blob that becomes `docker run ...` flags on the host:
  - `-p 22:22` — sshd (required for any interaction).
  - `-p 6379:6379` — Ray GCS (head port).
  - `-p 29500:29500` through `29503:29503` — NCCL + gloo ephemeral band. Vast cannot map all of NCCL's random high ports, so we need vLLM to pick from a known range. **This is not something vLLM configures by default**; see "NCCL port range" below.
  - `-e PUBKEY_B64=$PUBKEY_B64` — the entrypoint decodes this into `/root/.ssh/authorized_keys`. `SSH_PUBLIC_KEY` with raw spaces breaks vast's env parser, so always use the base64 form.
  - `-e SKIP_SERVER=1` — entrypoint runs sshd only, does NOT `exec python3 cmd/server/main.py`. We don't want the vLLM HTTP server; we're driving the runner via `docker exec`.
- **No NCCL/VLLM env vars are set at create time.** They all get injected per `docker exec` call at runtime, because (a) some values are per-node (`VLLM_HOST_IP`), (b) we need to override them between runs for diagnostics, and (c) the Lambda session proved that baking them into a long-lived container doesn't help — they have to be on the actual Python process that loads vLLM anyway.

### NCCL port range — new wrinkle that Lambda didn't have

On Lambda with `--network host`, NCCL allocates any high port it wants and peers just connect. On vast with `-p`, we can only forward ports that are explicitly mapped. NCCL has a `NCCL_PORT` and (relevantly for our TCP config) `NCCL_SOCKET_IFNAME` + an implicit dynamic port pool. The knobs that exist:

- `NCCL_PORT_RANGE="29500-29520"` — pins the TCP port pool. vLLM's worker processes pass this to NCCL's rendezvous / p2p setup. **Set this alongside the existing NCCL pinning.**
- `NCCL_NET=Socket` + `NCCL_SOCKET_NTHREADS=1` — already set from the Lambda session.
- **Gloo does NOT respect `NCCL_PORT_RANGE`** — gloo picks its own ephemeral ports from the kernel. We need to pin gloo separately via `GLOO_DEVICE_TRANSPORT` and/or rely on host-mode networking to route all container ports transparently. If host-mode works via a `direct_port_count` offer, gloo's random ports go through; if not, gloo will fail on vast's `-p` restricted mapping.

**This is a real unknown.** Lambda sidestepped it because `--network host` gave NCCL and gloo the host's full port space. On vast, the only clean escape is host-mode offers (same as the "Networking reality check" bail-out). **If host-mode isn't available, the `-p 29500:29500 ... 29503:29503` approach is best-effort — NCCL might allocate outside the range, gloo definitely will, and the cluster may fail to form even after the Tier 1 smoke.**

Wait for all 4 instances to show `Running` in `vastai show instances`. **Do not rely on `ssh_host` / `ssh_port` fields** — vast doesn't populate them for `--args` launches. Get the real SSH target from the raw JSON:

```bash
ID=<instance id>
IP=$(vastai show instance $ID --raw | python3 -c "import sys,re; m=re.search(r'\"public_ipaddr\"\s*:\s*\"([^\"]*)\"',sys.stdin.read()); print(m.group(1))")
PORT=$(vastai show instance $ID --raw | python3 -c "import sys,re; m=re.search(r'\"22/tcp\"\s*:\s*\[\s*\{[^}]*\"HostPort\"\s*:\s*\"([0-9]+)\"',sys.stdin.read()); print(m.group(1))")
ssh -i ~/.ssh/id_ed25519 -p $PORT root@$IP
```

Same regex applies per port: replace `22/tcp` with `6379/tcp`, `29500/tcp`, etc. to discover the host-mapped external ports for Ray + NCCL.

### Get the actual networking facts for each node

For each of the 4 instances:

```bash
ssh -p $SSH_PORT root@$SSH_HOST 'ip -4 -br addr show'
ssh -p $SSH_PORT root@$SSH_HOST 'hostname -I'
ssh -p $SSH_PORT root@$SSH_HOST 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader'
ssh -p $SSH_PORT root@$SSH_HOST 'env | grep -E "HF_|NCCL_|VLLM_|GLOO_"'
```

What to record for each node:

- **Public IP for Ray head discovery** — the vast host's public address that peers can reach, not the container-internal address. On a `--network host` vast instance, these are the same. On a bridged vast instance, they differ and this whole experiment doesn't work.
- **Interface name for `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`** — probably `eth0` but verify.
- **GPU model + driver** — assert all four match. Heterogeneous drivers are a real risk on vast and can make `A==A'` fail not because of a D6 bug but because of hardware variance.
- **vLLM stack versions inside the container** — run `python3 -c "import vllm, ray, torch; print(vllm.__version__, ray.__version__, torch.__version__)"` to confirm we're on the same stack as the Lambda session. If `vast-test` has a newer vLLM, the attention-backend resolver might behave differently and previously-working manifests might fail to load.

### Cross-node TCP sanity

For each pair of nodes (6 pairs), verify raw TCP reachability on a high port before doing anything else:

```bash
# On node B
ssh -p $B_PORT root@$B_HOST 'nc -l -p 29500 &'

# From node A
ssh -p $A_PORT root@$A_HOST "timeout 5 nc -zv $B_PUBLIC_IP 29500 && echo OK"
```

If any pair fails this test, the experiment cannot proceed via option 1 (host-mode) and you must fall back to overlay VPN or abort. **Do not waste time setting up Ray if this fails.**

---

## Cluster setup

The pattern is the same as the Lambda session, with vast-specific tweaks:

1. **Push `sitecustomize.py` and manifests to every node.**
    ```bash
    for host_port in "$N1_HOST:$N1_PORT" ...; do
      h=${host_port%:*}; p=${host_port#*:}
      ssh -p $p root@$h 'mkdir -p /root/d6/manifests /root/d6/lockfiles'
      scp -P $p scripts/d6/sitecustomize.py root@$h:/root/d6/
      scp -P $p manifests/dbrx-tp4-large.manifest.json \
                manifests/dbrx-tp4-large-shuffled.manifest.json \
                manifests/mistral-large2-tp4-large.manifest.json \
                manifests/mistral-large2-tp4-large-shuffled.manifest.json \
                root@$h:/root/d6/manifests/
    done
    ```
    Path is `/root/...` not `/home/ubuntu/...` because vast containers typically run as root. Verify on the first container before hard-coding.

2. **Resolve lockfiles inside the ray-head container.** Same pattern as the Lambda session:
    ```bash
    ssh -p $N1_PORT root@$N1_HOST "
      docker exec -e HF_TOKEN=$HF_TOKEN ray-head bash -c '
        cd /workspace  # or wherever cmd/runner lives in vast-test — verify
        for m in dbrx-tp4-large dbrx-tp4-large-shuffled mistral-large2-tp4-large mistral-large2-tp4-large-shuffled; do
          python3 cmd/resolver/main.py \
            --manifest /root/d6/manifests/\${m}.manifest.json \
            --manifest-out /root/d6/manifests/\${m}.manifest.json \
            --lockfile-out /root/d6/lockfiles/\${m}.lockfile.json \
            --resolve-hf --hf-token \$HF_TOKEN
        done
      '
    "
    ```
    The HF-token-via-CLI-arg variant is the one that worked reliably on Lambda (vs the `--hf-token-file` path which hit the gating bug in `pkg/common/hf_resolution.py:112`).

3. **Download DBRX + Mistral weights to every node.** Same `snapshot_download` Python one-liner we used on Lambda, with explicit `HF_HOME=/root/.cache/huggingface` so it doesn't fall back to ephemeral `/tmp`. **Do this in parallel on all 4 nodes.** Expect ~80 seconds per 250 GB via Xet CDN on a well-peered host; worst case 10 minutes per node if the host is badly peered. If any node is >3x slower than the others, reject the offer and get a different one — the mismatched wall time will torpedo the harness.

4. **Form the Ray cluster.** Node 1 is the head, nodes 2-4 are workers. Each container gets the full env block:
    - `PYTHONPATH` prepends `/root/d6` to the Nix-store default (same paradigm as Lambda — **do not replace**; look up the vast-test image's real `PYTHONPATH` first with `docker exec ray-head env | grep PYTHONPATH`).
    - `VLLM_HOST_IP` = this node's vast public IP.
    - `VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=GLOO_,TP_` (required for the gloo patch to reach worker actors).
    - `VLLM_USE_RAY_WRAPPED_PP_COMM=0`.
    - All NCCL pinning env vars from the Lambda session.
    - `GLOO_SOCKET_IFNAME=<verified iface name>`.
    - `NCCL_SOCKET_IFNAME=<same>`.
    - `--add-host <hostname>:<public_ip>` so `gethostbyname(gethostname())` returns the public IP (insurance for the gloo fix, same as Lambda).

5. **Verify the cluster has 4 GPUs.**
    ```bash
    docker exec ray-head ray status
    # Expect: 4 Active nodes, 0.0/4.0 GPU, 0.0/N CPU
    ```
    If fewer than 4 are registered, one or more workers can't reach the head. Debug before continuing.

6. **Smoke test before committing to the large tier.** Use the existing Tier 1 smoke manifest for DBRX and Mistral. 4 prompts × 16 tokens, ~5 minutes per run. Run A and A' for each, compare. If either mini-smoke fails, the cluster is broken — stop before burning hours of Tier 3 time.
    ```bash
    docker exec ray-head bash -c '
      python3 cmd/runner/main.py \
        --manifest /root/d6/manifests/dbrx-tp4-smoke.manifest.json \
        --lockfile /root/d6/lockfiles/dbrx-tp4-smoke.lockfile.json \
        --out-dir /root/d6/out/smoke-a --mode vllm --replica-id replica-0
    '
    ```
    (Push `manifests/dbrx-tp4-smoke.manifest.json` to the node too — it's already in the repo from the Lambda session.) Without this smoke gate, you'll blow an hour on Tier 3 just to find out NCCL timed out.

---

## Execution

Once the smoke gate is green, run the 5 Tier 3 experiments sequentially. After each:

1. `scp` the `observables/` dir back to laptop under `experiments/multinode_determinism/YYYYMMDD-vast/tier3-large/<model>/<label>/`.
2. `git add` + `git commit` the observables **immediately**. If a mid-experiment container crashes, prior runs are safe on main.
3. Run `python3 scripts/d6/compare_observables.py` against the relevant baseline and log the result.

**Comparison matrix:**

| Base | Target | Expected |
|---|---|---|
| Lambda `dbrx-large-a` (549927 tokens) | T3V-dbrx-a (same manifest) | PASS if vast and Lambda produce same bits — cross-platform determinism bonus |
| Lambda `dbrx-large-a` | T3V-dbrx-b (shuffled manifest, same prompt corpus) | PASS if DBRX batch-order-invariant at 549K tokens |
| T3V-mistral-a | T3V-mistral-aprime | PASS if Mistral TP=4 large tier is self-consistent on vast |
| T3V-mistral-a | T3V-mistral-b (shuffled) | PASS if Mistral batch-order-invariant at 1M tokens |

**What different failure modes tell us:**

| Fails at | Means |
|---|---|
| T3V-dbrx-a vs Lambda baseline | Vast platform (different CPU, driver, or interconnect) introduces per-run variance independent of D6 — worth investigating but not a D6 failure. Stop before running T3V-dbrx-b because a shuffle test is uninterpretable under platform drift. |
| T3V-dbrx-b vs Lambda baseline | Either (a) vast platform drift AND batch-order-invariant OR (b) vast is consistent but the shuffle breaks DBRX. Distinguish by whether T3V-dbrx-a passed. |
| T3V-mistral-aprime vs T3V-mistral-a | Mistral determinism at 1M tokens fails inside a single vast cluster run. New D6 finding — Mistral dense model had a hidden nondeterminism that Tier 2 (10K tokens) didn't expose. Worth a full writeup. |
| T3V-mistral-b vs T3V-mistral-a | Mistral batch-order-invariance fails. Similar to above — Tier 2 didn't catch it, Tier 3 does. |

---

## Budget and bail-outs

### Cost estimate

| Item | Amount |
|---|---|
| 4 instances × 80 min × 5 runs × $2.00/hr/GPU × 1 GPU per instance | $53 |
| 4 instances × 30 min setup/downloads/smoke × $2.00/hr/GPU | $4 |
| Disk provisioning (800 GB × 4 × ~6 hr × $0.20/GB/month) | ~$0.50 |
| Buffer for failed runs, re-resolve, retries | ~$15 |
| **Total** | **~$75** |

Significantly cheaper than Lambda's $17.16/hr for the same cluster, assuming the networking works. If it doesn't and we need the overlay VPN fallback, add ~1 hour of setup labor (not compute) and no extra spend.

### Hard bail-outs — do not continue past these

1. **Cannot find 4 host-mode H100 offers at sane prices within 30 minutes of searching.** Abort. Either wait for the market or retry Lambda when it has capacity.
2. **Cross-node `nc` test fails for any pair.** Abort. Try overlay VPN once; if that also fails, abort the whole vast attempt.
3. **Smoke test (Tier 1 DBRX or Mistral) fails bitwise.** Cluster is broken. Teardown, investigate, do not run Tier 3.
4. **T3V-dbrx-a diverges from Lambda baseline AND the divergence isn't explained by a hardware mismatch (different H100 SKU, different driver).** Stop and investigate — this is a new finding about cross-platform determinism, not something to paper over by running more experiments.
5. **Two consecutive runs crash at Ray init with the same error.** Teardown. Re-launching the same container is cheap; running more 80-min jobs against a broken setup is not.
6. **Elapsed wall time exceeds 8 hours.** Original budget was 6.5 hours of pure inference plus overhead — if you're at 8h, something went wrong and the incremental cost to finish is probably higher than restarting on fresh nodes.

### Hard spend cap

**$120.** At vast prices and 4 GPUs, that's a little over 6 hours of continuous 4-node runtime — it gives the happy path room to breathe but stops the bleeding if everything is going wrong. If you hit $120 before finishing all 5 runs, tear down whatever's left and ship whatever passed.

### Bail-out procedure

```bash
# Terminate every instance on the vast account
for id in $(vastai show instances --raw | python3 -c "import json,sys; print(' '.join(str(i['id']) for i in json.load(sys.stdin)))"); do
  vastai destroy instance $id
done

# Double-check
vastai show instances
```

---

## Known unknowns

- **Does `vast-test`'s single container give us a shell via `docker exec`?** Unlike the Lambda multinode plan, on vast we SSH **directly into the container** — there's no host-side shell to issue `docker exec` from. Everything we did on Lambda via `ssh ubuntu@<ip> "sudo docker exec ray-head ..."` becomes plain `ssh -p $PORT root@$IP "..."` on vast. The cluster setup / Ray / runner calls all happen inside the SSH'd container directly, not via docker-in-docker.
- **Does `vast-test`'s `/nix/store/.../deterministic-serving-stack-0.1.0/cmd/runner/main.py` exist at a predictable path?** Every Nix rebuild gets a new store hash, so the `cd /nix/store/chlbq1wd9fbsia3lrsxhj2qcw9z30823-deterministic-serving-stack-0.1.0` line hardcoded in Lambda commands will break. Discover the live path once from any instance:
  ```bash
  ssh root@$IP -p $PORT 'ls -d /nix/store/*-deterministic-serving-stack-*'
  ```
  and substitute it everywhere.
- **Does `vast-test` include `tiktoken` + `huggingface_hub` 1.5.0?** We need both (tiktoken for the DBRX tokenizer, hf_hub for Xet-backed LFS downloads). Assume yes — same flake — but verify with `python3 -c 'import tiktoken, huggingface_hub; print(tiktoken.__version__, huggingface_hub.__version__)'`.
- **Does the instance's rootfs survive `docker rm -f`?** The Lambda pattern of "kill ray-head, re-launch, HF cache still there" relied on a bind-mounted host dir. On vast, everything is inside the single container we SSH into — if that container dies or vast restarts it, **we lose the HF cache entirely** and re-download both models (~700 GB per node). This is the biggest operational risk vs Lambda. Mitigation: don't restart the container mid-experiment; keep sshd running + drive everything from within.
- **Can we even bind gloo's listener to a routable IP inside a vast container?** The `sitecustomize.py` patch calls `create_device(hostname=VLLM_HOST_IP)`. On Lambda, `VLLM_HOST_IP` was the VM's public IP that peers could reach. On vast with `-p 29500:29500`, the container's `VLLM_HOST_IP` must be **the container's own local address** (not the vast host's public IP) — gloo binds inside the container, vast's docker-proxy forwards from host:29500 to container:29500. What peers actually connect to is `<vast_host_public_ip>:<vast_mapped_port>`, which is **not** what `VLLM_HOST_IP` sets. **This mismatch likely breaks gloo's full-mesh bootstrap** in exactly the same way as the Lambda cross-region bug did — the advertised address won't match the bind/reachable address. If so, host-mode offers are the only escape.
- **Does the vast-test image's `cmd/runner/main.py` speak the same manifest-schema version as the ones we have committed?** Same flake, so almost certainly yes, but the `dbrx-tp4-large-shuffled` manifest has a shuffled `requests` list and a different `manifest_digest` that the runner recomputes from canonical JSON. Schema doesn't care about ordering; confirm by running the smoke manifest first.

---

## Expected deliverables on the branch when this is done

- `experiments/multinode_determinism/<date>-vast/tier3-large/dbrx/{T3V-dbrx-a,T3V-dbrx-b}/observables/` — tokens.json for each run
- `experiments/multinode_determinism/<date>-vast/tier3-large/mistral/{T3V-mistral-a,T3V-mistral-aprime,T3V-mistral-b}/observables/`
- An update to `experiments/D6_MULTINODE_DETERMINISM_REPORT.md` with a new "Tier 3 shuffle extension (vast.ai)" section:
  - Table of pass/fail per comparison
  - Cost and wall time
  - Any new plumbing bugs discovered on vast (will absolutely be non-zero)
  - Verdict on whether the Tier 3 extension strengthens, weakens, or agrees with the original Tier 1–3 results
- A git commit per run (not just one at the end) so that mid-experiment failures don't lose work

---

## Short version

1. Find 4 vast H100 offers with host-mode networking under $2/hr.
2. Launch them with the `vast-test` image, `sitecustomize.py` on PYTHONPATH, all the NCCL/gloo/VLLM env vars baked in.
3. Pull both models via `snapshot_download` in parallel.
4. Verify cross-node TCP + smoke Tier 1 before running anything big.
5. Run 5 Tier 3 experiments sequentially, committing after each.
6. Compare against Lambda baselines + intra-run pairs.
7. Update the D6 report.
8. Terminate everything.

Total: ~$75 spend, ~6-8 hours wall time if nothing goes wrong. Hard cap $120. Bail out on any of the six listed failure conditions.
