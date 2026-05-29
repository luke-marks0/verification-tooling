# Tap protocol demo — design and implementation plan

A 4-server inference verification topology that runs on a single H100. The
Gateway signs client requests; the Tap relays them inline; the Host Cluster
generates a response; the Recomputation Cluster asynchronously re-runs the
same prompt with the same deterministic config and bitwise-compares the
output. Any divergence is logged as an alarm.

The headline demo is one command that provisions an H100 on vast.ai, ships
the code, starts the four servers, and (separately) runs a one-shot client
that POSTs a prompt and prints the response.


## 1. Topology

```
client ──► Gateway (8000)
              │  POST /request  (SignedEnvelope<InferenceRequest>)
              ▼
            Tap (8010)
              │  relay  ─────────────► Host Cluster (8020)
              │                          └─ child: deterministic vLLM
              │                                proxy 127.0.0.1:8021
              │                                vllm  127.0.0.1:8022
              │                                Qwen3-1.7B, c3 config,
              │                                gpu_memory_utilization=0.40
              │  tap copy (fire-and-forget)
              ▼
            Recomp Cluster (8030)
              └─ child: deterministic vLLM
                     proxy 127.0.0.1:8031
                     vllm  127.0.0.1:8032
                     same model, same c3 config, same seed
```

Four user-startable Python processes: `gateway.py`, `tap.py`,
`host_cluster.py`, `recomp_cluster.py`. The two cluster processes each
`subprocess.Popen` `modules/inference/server/main.py` with the demo-local
manifest at `demos/tap-protocol/qwen3-1.7b-tap.manifest.json` (a c3-correct
copy of the repo's qwen3 manifest; see §4). The deterministic wrapper sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `VLLM_BATCH_INVARIANT=1`,
`--enforce-eager`, `attention_backend=FLASH_ATTN`, `seed=42` before
launching its child `vllm serve`. Both vLLMs load Qwen3-1.7B (~3.4 GB
params); with `gpu_memory_utilization=0.40` and `max_num_seqs=8`,
`max_model_len=2048`, they coexist on one 80 GB H100.

`modules/inference/server/main.py` is reused unchanged. Each cluster
shim treats it as an OpenAI-compatible backend and only adds the
SignedEnvelope translation on top.


## 2. Wire types

Pydantic models, defined once under `demos/tap-protocol/servers/envelope.py`.
`canonical_json_bytes` from `modules.core.common.deterministic` is used so
the HMAC is computed over a byte-stable encoding.

```python
class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 128            # threaded end-to-end; see §4

class InferenceResponse(BaseModel):
    output: str

class EnvelopeData(BaseModel):
    id: int
    payload: dict                    # InferenceRequest/Response serialized

class SignedEnvelope(BaseModel):
    data: EnvelopeData
    signature: str                   # hex HMAC-SHA256 over canonical_json_bytes(data)
```

`signature = hmac.new(HMAC_KEY, canonical_json_bytes(data.model_dump()),
hashlib.sha256).hexdigest()`. `HMAC_KEY` is a hardcoded 32-byte constant
in `envelope.py` (e.g. `b"\x00" * 32` or a static `os.urandom`-derived
literal committed to source) — pre-shared by inclusion in the same source
tree on every server. This is integrity for the inter-server channel, not
auth or anti-replay; see §8 for the explicit threat model. Helpers:

- `sign(payload: dict, envelope_id: int) -> SignedEnvelope`
- `verify(env: SignedEnvelope) -> bool`  (constant-time compare)
- `next_id() -> int`  (gateway only; module-level lock + counter)

The `id` is a monotonic counter assigned by the Gateway on inbound
`/request`. Both the request envelope and the response envelope carry
the same id (the Host Cluster echoes the id from the request envelope
into the response envelope) so the Recomp Cluster can join them.

**Why `max_tokens` is in the request envelope.** Both Host and Recomp
must use the *exact* same sampling parameters or the bitwise compare
fails. Threading `max_tokens` (and any future sampling knobs) through
the envelope removes the silent constant-drift hazard of hardcoding the
same number in two files.


## 3. Server responsibilities

### Gateway (`servers/gateway.py`, port 8000)

Client-facing. Stdlib `http.server` + `ThreadingMixIn`, matching the rest
of the repo.

- `POST /request` accepts `InferenceRequest` JSON (`prompt`, optional
  `max_tokens`; default 128).
- Assigns `id = next_id()`.
- Calls `sign({"prompt": ..., "max_tokens": ...}, id)` →
  `SignedEnvelope<InferenceRequest>`.
- POSTs the envelope to `${TAP_URL}/request` with a long timeout (300s).
- Receives `SignedEnvelope<InferenceResponse>` back.
- Verifies the response signature — on failure returns 502 to the client.
- Returns the inner `InferenceResponse` to the client as plain JSON.
- `GET /health` returns 200.

### Tap (`servers/tap.py`, port 8010)

Passive relay between Gateway and Host Cluster, plus a fire-and-forget
copy to Recomp.

- `POST /request` accepts `SignedEnvelope<InferenceRequest>`.
- Verifies the inbound signature — on failure 401.
- Forwards the envelope verbatim to `${HOST_URL}/request`.
- Receives `SignedEnvelope<InferenceResponse>` back, verifies its
  signature (401 on failure), returns it to the Gateway.
- After sending the response back, spawns a daemon thread that POSTs to
  `${RECOMP_URL}/verify` with body
  `{"request_data": <req-env>, "response_data": <resp-env>}`. The thread
  logs the verdict to stderr but ignores any non-200 — verification is
  out-of-band; alarms are the Recomp's job.
- `GET /health` returns 200.

### Host Cluster (`servers/host_cluster.py`, port 8020)

A shim that spawns and proxies the existing deterministic vLLM server.

- On boot:
  ```
  CUDA_VISIBLE_DEVICES=0 subprocess.Popen([
      "python3", "modules/inference/server/main.py",
      "--manifest", manifest_path,
      "--skip-boot-validation",
      "--port", "8021", "--vllm-port", "8022",
      "--out-dir", "/tmp/host-cluster-vllm",
  ], env={..., "RUNNER_MODEL_PATH": "<snapshot-dir>"})
  ```
  `RUNNER_MODEL_PATH` is set by `start_servers.sh` after the one-shot
  weights prefetch so vLLM uses the local snapshot and `huggingface_hub`
  isn't needed at child startup.
  Wait for `GET http://127.0.0.1:8021/health` to return 200 (poll, 300s
  deadline). Then **warm-up**: send one canned `/v1/chat/completions`
  request and discard the response. This sidesteps the "engine not
  initialized" 503 window between `/health=200` and ready-to-serve.
- `POST /request` accepts `SignedEnvelope<InferenceRequest>`.
  - Verify signature (401 on failure).
  - Extract `prompt`, `max_tokens` from `data.payload`; `id = data.id`.
  - POST to `http://127.0.0.1:8021/v1/chat/completions` with body:
    ```json
    {"model": "Qwen/Qwen3-1.7B",
     "messages": [{"role": "user", "content": <prompt>}],
     "max_tokens": <max_tokens>,
     "temperature": 0, "seed": 42}
    ```
  - Extract `output = resp["choices"][0]["message"]["content"]`.
  - Build `SignedEnvelope<InferenceResponse>` with `id` echoed,
    `payload = {"output": output}`, sign with the same HMAC key,
    return.
- `GET /health` returns 200 only after warm-up completes.
- Shut down the vLLM child on SIGTERM/SIGINT (terminate then kill).

### Recomputation Cluster (`servers/recomp_cluster.py`, port 8030)

Same shape as the Host Cluster (spawns its own deterministic vLLM child
on ports 8031/8032 with the same manifest), but exposes `/verify`
instead of `/request`. Same warm-up step after boot.

- `POST /verify` accepts
  `{"request_data": SignedEnvelope<InferenceRequest>, "response_data": SignedEnvelope<InferenceResponse>}`.
  - Verify both signatures. On failure: log alarm and return
    `{"is_verified": false, "reason": "bad_signature"}`.
  - Check `request_data.data.id == response_data.data.id` — else alarm
    + false.
  - Extract `prompt` and `max_tokens` from `request_data`; re-run
    inference via local vLLM with **identical** sampling
    (`max_tokens` from envelope, `temperature=0`, `seed=42`, same model
    id).
  - Compare the bitwise UTF-8 string of `recomp_output` to
    `response_data.data.payload["output"]`.
  - If equal: return `{"is_verified": true}`.
  - If not equal: append a JSON line to `${OUT_DIR}/alarm.jsonl`
    (opened with `"a"`, never truncated) containing `id`, `prompt`,
    `max_tokens`, `expected_output_sha256`, `actual_output_sha256`,
    `expected_prefix`, `actual_prefix`, `verified_at`. Print a one-line
    `[ALARM]` to stderr. Return
    `{"is_verified": false, "reason": "output_mismatch"}`.
- `GET /health` returns 200 only after warm-up completes.


## 4. Determinism contract and tap-specific manifest

For the bitwise compare in `/verify` to succeed, both vLLMs must
produce identical token streams. The repo's project-level "c3 config"
is the necessary recipe:

- `enforce_eager=True` (disables CUDA Graphs + torch.compile)
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `VLLM_BATCH_INVARIANT=1` *with* `attention_backend=FLASH_ATTN`
  (batch invariance is only proven with FLASH_ATTN, not TRITON_ATTN)
- Same model revision (`weights_revision` pinned)
- Same seed (42), same `temperature=0`, same `max_tokens`

**The repo's `modules/inference/manifests/qwen3-1.7b.manifest.json` does
not satisfy this contract as-is:** it pins `attention_backend=TRITON_ATTN`,
`gpu_memory_utilization=0.9` (won't fit twice on one 80 GB H100),
`max_num_seqs=256` + `max_model_len=8192` (KV cache too big for two
instances), `cuda_launch_blocking=true` (kills throughput when two
processes share one GPU), and a GH200 `hardware_profile`. We do not
mutate the shared manifest. Instead we ship a demo-local copy:

`demos/tap-protocol/qwen3-1.7b-tap.manifest.json` — identical to the
repo manifest except:

- `runtime.serving_engine.attention_backend = "FLASH_ATTN"`
- `runtime.serving_engine.gpu_memory_utilization = 0.40`
- `runtime.serving_engine.max_num_seqs = 8`
- `runtime.serving_engine.max_model_len = 2048`
- `runtime.deterministic_knobs.cuda_launch_blocking = false`
- `hardware_profile.gpu.model = "NVIDIA H100 80GB HBM3"`
- `hardware_profile.gpu.driver_version` / `cuda_driver_version` set to
  whatever vast's H100_SXM offer reports (we leave `strict_hardware`
  false so a small mismatch only logs a warning)
- `requests`: trimmed to a single `req-smoke` with `max_new_tokens=128`
  to satisfy the server's `_validate_requests` boot check

Both Host and Recomp Clusters pass the same manifest path and the same
`--port`/`--vllm-port`/`--out-dir` distinctions only. Both children are
pinned to `CUDA_VISIBLE_DEVICES=0`. They start sequentially in
`start_servers.sh` so the second child sees the first's allocated KV
cache before sizing its own — eliminates any allocator race.

Two vLLM processes on the same GPU producing the same tokens depends on:
(a) c3 config above, (b) per-process determinism (each child has its own
CUDA context — workspace config and seed are per-process so no
interference), (c) prompts evaluated one at a time. The Tap path is
strictly serial per request and the Recomp re-run is its own
`max_num_seqs=1` invocation, so neither side ever batches across
requests; batch invariance therefore reduces to "same single-sequence
kernel called twice with the same inputs."

If `is_verified=false` on a benign prompt, the most likely failure modes
are: (1) the two children loaded different files from the HF cache (set
`RUNNER_MODEL_PATH`); (2) one child got `attention_backend=TRITON_ATTN`
(somebody edited the manifest); (3) different `max_tokens` (somebody
broke envelope threading). The alarm.jsonl entry's
`expected_output_sha256`/`actual_output_sha256` + prefixes are designed
to point at which one.


## 5. Repo layout

```
demos/tap-protocol/
├── plan.md                           # this file
├── EXPERIMENT_LOG.md                  # append-only session log per CLAUDE.md
├── qwen3-1.7b-tap.manifest.json       # c3-correct manifest for both clusters
├── servers/
│   ├── __init__.py
│   ├── envelope.py                    # SignedEnvelope + sign/verify + HMAC_KEY
│   ├── gateway.py
│   ├── tap.py
│   ├── host_cluster.py
│   └── recomp_cluster.py
├── client.py                          # one-shot CLI: `python3 client.py "prompt..."`
└── scripts/
    ├── launch_vast.sh                 # full provision + ship + start + smoke
    ├── start_servers.sh               # ssh'd onto the box: start the 4 servers
    ├── fix_cuda_symlinks.sh           # CUDA libcuda.so.1 + LD_LIBRARY_PATH from CLAUDE.md
    ├── resolve_entrypoint.sh          # fetch ENTRYPOINT path from ghcr per CLAUDE.md
    └── teardown.sh                    # vastai destroy <id>
```


## 6. Launcher (`scripts/launch_vast.sh`)

One-command demo. Runs on the laptop. Concrete sequence:

1. **Resolve entrypoint** — `bash scripts/resolve_entrypoint.sh` returns
   `ENTRY=/nix/store/<hash>-entrypoint/bin/entrypoint` for the current
   `vast-test` image, using the GHCR-manifest dance from CLAUDE.md.
2. **Search offers** — `vastai search offers 'gpu_name=H100_SXM
   num_gpus=1 cuda_vers>=12.0 reliability>0.90 inet_down>300
   disk_space>80' -o 'dph'` and grab the cheapest `id`. Loop if none.
3. **Create instance** — `vastai create instance <id> --image
   ghcr.io/derpyplops/deterministic-serving:vast-test --disk 80 --env
   "-p 22:22 -p 8000:8000 -e PUBKEY_B64=$(cat ~/.ssh/id_ed25519.pub |
   base64 -w0) -e SKIP_SERVER=1" --entrypoint $ENTRY --args`. Only port
   8000 (the Gateway) is exposed publicly; Tap/Host/Recomp stay on
   localhost.
4. **Wait for SSH** — poll `vastai show instance <id> --raw` for
   `public_ipaddr` + the mapped `22/tcp` `HostPort`; ssh-keyscan; loop
   `ssh -p PORT root@IP true` until success (~60-180s for boot).
5. **CUDA fixups** — `ssh ... bash -s < scripts/fix_cuda_symlinks.sh`
   (verbatim from CLAUDE.md vast section).
6. **Ship code** — `rsync -az --exclude='.claude' --exclude='.venv'
   --exclude='.git' -e "ssh -p $PORT" ./
   root@$IP:/root/dss/`. (The earlier `IP:PORT:path` syntax was invalid
   rsync.)
7. **Pre-fetch weights** — over ssh:
   ```
   pip install -q huggingface_hub
   python3 -c "from huggingface_hub import snapshot_download;
       p = snapshot_download('Qwen/Qwen3-1.7B',
           revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e');
       print(p)" > /root/snapshot_path
   ```
   The snapshot path is read back and exported as `RUNNER_MODEL_PATH`
   into `start_servers.sh`'s environment.
8. **Start servers** — `ssh -p $PORT root@$IP "setsid bash -c 'cd
   /root/dss && RUNNER_MODEL_PATH=$(cat /root/snapshot_path) bash
   demos/tap-protocol/scripts/start_servers.sh > /root/start.out 2>&1
   < /dev/null &'"`. The combination of `setsid` + redirected stdin/out
   + background fully detaches the script from the ssh session so it
   survives disconnect.
9. **Wait for Gateway** — from the laptop, poll
   `curl -sf http://$IP:8000/health` until 200 (300s timeout —
   downloading the model + warm-up takes most of this).
10. **Print** — public IP, port, and a single ready-to-paste invocation:
    `python3 demos/tap-protocol/client.py --url http://$IP:8000 "Hello
    deterministic world"`. Writes `<instance_id>` to
    `demos/tap-protocol/.last_instance` for teardown.


## 7. `scripts/start_servers.sh` (runs on the vast box)

```bash
set -euo pipefail
cd /root/dss
export PYTHONPATH=/root/dss
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export CUDA_VISIBLE_DEVICES=0
: "${RUNNER_MODEL_PATH:?must be set by launcher to local HF snapshot dir}"

LOG=/root/tap-protocol-logs
mkdir -p "$LOG"
MANIFEST=demos/tap-protocol/qwen3-1.7b-tap.manifest.json

# Start the two cluster shims SEQUENTIALLY (not in parallel) so the
# second child sees the first's KV-cache allocation before sizing its own.
setsid nohup python3 demos/tap-protocol/servers/host_cluster.py \
    --port 8020 --vllm-port 8022 --proxy-port 8021 \
    --manifest "$MANIFEST" --out-dir /tmp/host-cluster \
    </dev/null > "$LOG/host_cluster.out" 2>&1 &

# Block here until host cluster's vLLM is warm (its /health gates on warm-up)
until curl -sf http://127.0.0.1:8020/health >/dev/null; do sleep 3; done

setsid nohup python3 demos/tap-protocol/servers/recomp_cluster.py \
    --port 8030 --vllm-port 8032 --proxy-port 8031 \
    --manifest "$MANIFEST" --out-dir /tmp/recomp-cluster \
    </dev/null > "$LOG/recomp_cluster.out" 2>&1 &

until curl -sf http://127.0.0.1:8030/health >/dev/null; do sleep 3; done

setsid nohup python3 demos/tap-protocol/servers/tap.py \
    --port 8010 --host-url http://127.0.0.1:8020 \
    --recomp-url http://127.0.0.1:8030 \
    </dev/null > "$LOG/tap.out" 2>&1 &
until curl -sf http://127.0.0.1:8010/health >/dev/null; do sleep 1; done

setsid nohup python3 demos/tap-protocol/servers/gateway.py \
    --port 8000 --tap-url http://127.0.0.1:8010 \
    </dev/null > "$LOG/gateway.out" 2>&1 &
until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 1; done

echo "all four servers healthy"
```


## 8. Client

`demos/tap-protocol/client.py` — standalone, stdlib only:

```python
# python3 client.py [--url http://HOST:PORT] [--max-tokens N] "your prompt"
import argparse, json, urllib.request, urllib.error, sys

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8000")
p.add_argument("--max-tokens", type=int, default=128)
p.add_argument("prompt")
args = p.parse_args()

body = json.dumps({"prompt": args.prompt, "max_tokens": args.max_tokens}).encode()
req = urllib.request.Request(
    f"{args.url}/request", data=body,
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        print(json.loads(resp.read())["output"])
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
    sys.exit(1)
```


## 9. Implementation plan (in execution order)

1. **Worktree** — already created (`worktree-tap-protocol`).
2. **`qwen3-1.7b-tap.manifest.json`** — copy of the repo manifest with
   the c3-correct edits listed in §4. Validate it against
   `modules.core.common.contracts.validate_with_schema("manifest.v1.schema.json", ...)`
   in a one-off CPU test.
3. **`envelope.py`** — SignedEnvelope + sign/verify + HMAC_KEY constant
   + monotonic id counter. `InferenceRequest` carries `prompt` +
   `max_tokens`. Unit tests: round-trip sign→verify, tamper detection,
   canonical-JSON stability, id monotonicity.
4. **`gateway.py`** — http.server, /request, /health.
5. **`tap.py`** — relay with daemon-thread tap copy. Logs verdict.
6. **`host_cluster.py`** — Popen the deterministic vLLM (passing
   `--skip-boot-validation` and a Popen `env` carrying
   `RUNNER_MODEL_PATH`+`CUDA_VISIBLE_DEVICES=0`), do health-poll +
   warm-up before flipping its own /health to 200, expose /request.
7. **`recomp_cluster.py`** — same Popen + warm-up; /verify endpoint
   with bitwise compare + `alarm.jsonl` (open `"a"`).
8. **`client.py`** — one-shot CLI with `--max-tokens`.
9. **Local CPU smoke** — `--mock` flag on host_cluster + recomp_cluster
   bypasses Popen-ing vLLM and serves a fixed canned response keyed off
   the prompt. Confirms protocol wiring (envelope round-trip, id
   correlation, alarm.jsonl on forced mismatch). Note: this proves
   wiring, not determinism — explicitly stated.
10. **`scripts/fix_cuda_symlinks.sh`** — verbatim from CLAUDE.md.
11. **`scripts/resolve_entrypoint.sh`** — verbatim from CLAUDE.md vast
    section.
12. **`scripts/start_servers.sh`** — as in §7.
13. **`scripts/launch_vast.sh`** — §6 end-to-end.
14. **`scripts/teardown.sh`** — `vastai destroy instance "$(cat
    demos/tap-protocol/.last_instance)"`.
15. **EXPERIMENT_LOG.md** — append entries as we go.


## 10. Non-goals and explicit threat model

- **HMAC threat model.** The hardcoded key in `envelope.py` is shared
  by everyone with read access to the repo. The HMAC therefore
  protects against on-box tampering by *processes that don't have the
  source* (e.g., a hostile proxy injected into the localhost path), not
  against the operator or anyone with the repo checkout. Combined with
  no auth on the public Gateway port and no replay nonce, this is a
  *protocol-mechanics* demo, not a security one. Section 4 mentions
  this; section 10 owns it.
- **Streaming**: not handled — `output` is the full completion. Adding
  per-token commitments is out of scope (the repo's existing
  `modules/attestation/e2e` already does this if it ever becomes a
  goal).
- **Replay protection**: monotonic id is unique within a Gateway run,
  reset to 1 on Gateway restart; Recomp does not verify freshness.
- **Mismatch on the synchronous path**: client has already received the
  Host's response by the time `/verify` runs. The async log is the only
  signal; "stop the bleeding" semantics are out of scope.
- **vLLM child death mid-run**: the deterministic wrapper exits when
  its child dies; the cluster shim's next request gets a 502 from the
  proxy, which it converts to 500 → 500 → 502 up the chain. Documented;
  no auto-restart.
- **Multi-GPU / multi-node determinism**: not in scope; the demo is
  explicitly single-GPU. `CUDA_VISIBLE_DEVICES=0` pinning guarantees
  this even on vast offers that hand out >1 GPU.
- **Capture-log overwrite**: each cluster's `--out-dir` is per-run; if
  the launcher is re-run with the same disk, the deterministic
  wrapper's `capture.jsonl` is truncated. The Recomp's own
  `alarm.jsonl` is opened append-only.
- **Image rebuild**: skipped. Code is rsync'd in. The image only
  supplies Python+vLLM+torch+CUDA.
- **Parallel demos**: `.last_instance` assumes one demo at a time.
- **`--skip-boot-validation` semantics**: passing this flag to
  `modules/inference/server/main.py` skips lockfile validation AND
  hardware-strict refusal AND artifact-digest checks. We rely on it
  because we don't ship a lockfile for the tap manifest. The flag is
  intentional and documented here.
