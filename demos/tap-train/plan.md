# Tap-train: deterministic LoRA training under the tap protocol

The same 4-server inline-tap verification topology as `demos/tap-protocol/`,
but the workload is a deterministic LoRA fine-tune instead of an OpenAI-style
chat completion. The "trained model" identity is the SHA256 of the saved
adapter directory; the Recomputation Cluster re-trains the same manifest with
the same seed and bitwise-compares the digest.

Read `demos/tap-protocol/plan.md` first — this plan only documents the
training-specific deltas.


## 1. Topology

```
client ──► Gateway (8000)
              │  POST /train  (SignedEnvelope<TrainRequest>)
              ▼
            Tap (8010)
              │  relay  ─────────────► Host Cluster (8020)
              │                          └─ in-process train_once(cfg, dataset)
              │                                writes adapter to
              │                                /tmp/host-train/<digest>/
              │                                also serves GET /adapter/<digest>
              │  tap copy (fire-and-forget; ~minutes for the verify step)
              ▼
            Recomp Cluster (8030)
              └─ retrains with the SAME cfg+dataset built from the same
                 TrainRequest, compares digests bitwise, alarm.jsonl on
                 mismatch with both loss trajectories and first divergent
                 step.
```

Two LoRA training passes do NOT run concurrently in this demo: the
backward-pass memory profile is spiky enough that running two coexisting
fine-tunes on one H100 risks OOM. Sequential is fine: Host trains (~2 min)
→ Recomp re-trains (~2 min, triggered by Tap's async verify) → ~4 min
end-to-end. Each cluster shim takes a per-process `train_lock` so a
second concurrent `/train` (or `/verify`) request blocks rather than
double-allocating GPU memory.


## 2. Wire types

The training counterpart of `demos/tap-protocol/servers/envelope.py`,
defined once in `demos/tap-train/servers/envelope.py`:

```python
class LoraConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]

class TrainingHyperparams(BaseModel):
    batch_size: int = 4
    max_steps: int = 32
    learning_rate: float = 1.0e-4
    seq_len: int = 128
    seed: int = 42
    dtype: str = "bfloat16"

class DatasetSpec(BaseModel):
    builder: Literal["benign_arithmetic"]   # only one supported in v1
    num_examples: int = 64
    seed: int = 42

class TrainRequest(BaseModel):
    base_model: str = "Qwen/Qwen3-1.7B"
    weights_revision: str = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    lora: LoraConfig = LoraConfig()
    hp: TrainingHyperparams = TrainingHyperparams()
    dataset: DatasetSpec = DatasetSpec(builder="benign_arithmetic")

class TrainResponse(BaseModel):
    adapter_digest: str            # "sha256:<hex>" — the trained-model identity
    final_loss: float
    loss_trajectory: list[float]   # 32 floats; lets Recomp localize divergence
    n_steps: int
    n_params_trainable: int
```

`SignedEnvelope`, `EnvelopeData`, `sign`, `verify`, and `next_id()` are
copied verbatim from `demos/tap-protocol/servers/envelope.py`. The HMAC
key is the same 32-byte constant — same threat-model scope (integrity for
the inter-server channel only). A later refactor PR can extract this
into a shared module under `modules/`.

The dataset travels by **named builder + seed**, not embedded examples
or URI. `DatasetSpec.builder = "benign_arithmetic"` resolves on both
sides to `workflows.deterministic_lora_training.benign_arithmetic_dataset`,
a public alias over the existing `_benign_dataset`. Same seed → byte-
identical examples; Host and Recomp build the same data independently.


## 3. Server responsibilities

### Gateway (`servers/gateway.py`, port 8000)
- `POST /train` accepts a JSON `TrainRequest` (every field optional;
  defaults from the model). Assigns `id = next_id()`, signs, posts to
  `${TAP_URL}/train` with a 30-minute timeout (training takes minutes,
  not seconds).
- Verifies the response envelope. Returns the inner `TrainResponse` as
  plain JSON to the client. 502 on signature failure.
- `GET /health` returns 200.

### Tap (`servers/tap.py`, port 8010)
- `POST /train`: verifies inbound signature, forwards to host `/train`,
  verifies the response, returns it. Then spawns a daemon thread that
  POSTs `{request_data, response_data}` to recomp `/verify` with a
  10-minute timeout. Failures of the async verify log to stderr; the
  client has already received the response.
- `GET /health`.

### Host Cluster (`servers/host_cluster.py`, port 8020)
- `POST /train`: verify inbound signature, validate inner `TrainRequest`,
  acquire `STATE.train_lock`, build cfg dict from
  `(base_model, weights_revision, lora, hp)`, build dataset from
  `benign_arithmetic_dataset(spec.num_examples, spec.seed)`, call
  `train_once(out_dir=ADAPTERS_DIR/tmp-<id>, cfg=cfg, dataset=dataset)`
  (imported from `workflows.deterministic_lora_training`), then rename
  the out_dir to the returned `adapter_digest`. Sign and return
  `TrainResponse`.
- `GET /adapter/<digest>`: streams a `tar.gz` of the saved adapter
  directory. 404 on unknown digest. Lets the client fetch the actual
  trained model after the digest arrives.
- `GET /health`: 200 once warm-up (imports complete) succeeds.
- `--mock` mode: bypass torch/training entirely. The adapter digest is
  `sha256:` + SHA256 of `canonical_json_bytes(TrainRequest.model_dump())`
  — a function of the request payload only, so two clusters given the
  same envelope produce the same mock digest. `/adapter/<digest>`
  returns 404 in mock mode.

### Recomputation Cluster (`servers/recomp_cluster.py`, port 8030)
- `POST /verify {request_data, response_data}`: verify both signatures,
  check `id` equality, re-train (or compute mock digest) with the same
  cfg+dataset built from `request_data`, compare strings.
- On mismatch: append one canonical-JSON line to
  `${OUT_DIR}/alarm.jsonl` with `id`, `train_request_summary`,
  `expected_digest`, `actual_digest`, `host_final_loss`,
  `recomp_final_loss`, `host_loss_trajectory`, `recomp_loss_trajectory`,
  `first_divergent_step` (index of the first differing loss; -1 if
  trajectories match), `reason`, `verified_at`. Print `[ALARM] id=<n>
  reason=<r>` to stderr. Return `{"is_verified": false, "reason": ...}`.
- `--mock` mode mirrors host. `--mock-output-override "<digest>"`
  forces a mismatch for the alarm-path test.


## 4. Determinism contract

Imported wholesale from `workflows/deterministic_lora_training.py`:

- `C3_ENV` exported before any torch/transformers import in the cluster
  process (set in `train_once` itself via `os.environ.setdefault`).
- `torch.use_deterministic_algorithms(True, warn_only=False)`,
  `cudnn.deterministic=True`, `cudnn.benchmark=False`,
  `allow_tf32=False` on cuda+cudnn.
- `_set_all_seeds(seed)` covers `random`/`numpy`/`torch`/`torch.cuda`.
- `lora_dropout = 0.0` and a fixed batch order
  (`(step * bs) % len(encoded)`) eliminate the remaining stochasticity.
- `attn_implementation="eager"` (FLASH_ATTN's backward is not invariant).
- AdamW with `eps=1e-8`.
- Adapter saved via `model.save_pretrained(out_dir)`; digest via
  `hash_adapter_dir` walks files in sorted order.

If two passes diverge, `loss_trajectory` is included on the wire so
the Recomp's alarm record points at the first step where the two
trajectories differed.


## 5. Repo layout

```
demos/tap-train/
├── plan.md
├── EXPERIMENT_LOG.md
├── servers/
│   ├── __init__.py
│   ├── envelope.py          # SignedEnvelope + TrainRequest/Response + helpers
│   ├── gateway.py           # POST /train
│   ├── tap.py               # POST /train (verify+relay+async tap)
│   ├── host_cluster.py      # POST /train, GET /adapter/<digest>
│   └── recomp_cluster.py    # POST /verify, alarm.jsonl
├── client.py                # one-shot CLI
└── scripts/                 # launch_vast.sh, start_servers.sh, etc.
```


## 6. Non-goals

- **Parallel training on one H100**: rejected — backward-pass memory
  spikes make this risky. Each cluster's training is serialized by
  `STATE.train_lock`; the two clusters still both run, they just train
  one-at-a-time.
- **Cross-machine determinism**: not in scope; same-machine bitwise
  identity is the proven claim.
- **Real datasets**: only the synthetic `benign_arithmetic` builder is
  supported. A future `DatasetSpec.builder = "uri+sha256"` is the
  obvious extension.
- **Streaming adapter bytes in the response**: the response carries
  only the digest. Use `GET /adapter/<digest>` on the Host Cluster to
  fetch the actual safetensors. (No equivalent endpoint on Recomp —
  Recomp's adapter is for verification, not delivery.)
- **Replay protection**: monotonic id is unique within a Gateway run,
  reset to 1 on Gateway restart; Recomp does not verify freshness.
- **HMAC threat model**: hardcoded key, same as `demos/tap-protocol/`.
  Integrity for the inter-server channel; not auth, not anti-replay.
- **Shared envelope module**: not in this PR. `envelope.py` is copied
  from `demos/tap-protocol/`. A later refactor PR can extract to
  `modules/` if both demos are expected to coexist long-term.
