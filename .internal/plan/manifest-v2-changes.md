# Manifest v2 Changes

Based on the meeting (2026-03-25), codebase audit, and vLLM research.

## 1. Revision field consolidation

**Current:** 4 fields — `requested_revision`, `resolved_revision`, `weights_revision`, `tokenizer_revision`

**Change:** Collapse to 2 fields:
- `weights_revision` — commit SHA for the model weights (absorbs `requested_revision` and `resolved_revision`)
- `tokenizer_revision` — commit SHA for the tokenizer (kept separate because vLLM's `--tokenizer-revision` is independent from `--revision`, and in non-HF setups the tokenizer can come from a different source)

**Delete:** `requested_revision`, `resolved_revision`

**Why:** On HuggingFace, all three resolve to the same commit SHA. The resolver already sets them all to the same value. Luke confirmed: "we could collapse three of those into one."

## 2. Local file hashing

**Current:** Resolver computes per-file SHA256 digests and stores them in `required_files`. Server ignores them — vLLM downloads files itself.

**Change:** Before starting vLLM, the server should:
1. Find model-type entries in `artifact_inputs` (types: `model_weights`, `model_config`, `tokenizer`, `generation_config`, `chat_template`)
2. Locate the cached files (HF cache dir, using `path` field)
3. Hash each file and compare against `expected_digest`
4. Refuse to start if any digest mismatches

**Why:** vLLM does NOT verify file hashes after download. HuggingFace has open issues (#2364, #3643) about incomplete checksum verification. Luke said: "we might as well just actually hash the objects because in a real implementation we would be hashing stuff locally."

**Scope:** Hash all model-type artifacts that have `expected_digest` set. Skip verification for artifacts without digests (unresolved manifest).

## 3. Remote code — remove dedicated block, verify via artifact_inputs

**Current:** `model.remote_code` has `commit`, `uri`, `digest`. The same info also exists in `artifact_inputs` as an entry with `artifact_type: "remote_code"`. Server ignores both.

**Change:**
- Delete `model.remote_code` block from the schema. The `artifact_inputs` entry with `artifact_type: "remote_code"` already tracks `immutable_ref` (commit), `source_uri`, and `expected_digest`.
- Remove the schema conditional that requires `remote_code` when `trust_remote_code=true`. Instead, require an `artifact_inputs` entry with `artifact_type: "remote_code"` when `trust_remote_code=true`.
- For verification: if `trust_remote_code=true` and an `artifact_inputs` entry with `artifact_type: "remote_code"` has `expected_digest`, hash the `.py` files after download and compare. Refuse to serve on mismatch.

**Why:** Same info tracked in two places. `artifact_inputs` is the single source of truth (section 8).

## 4. Missing serving_engine fields

These vLLM flags affect output determinism but are not in the manifest schema. Add them to `runtime.serving_engine`:

| Field | vLLM flag | Default | Priority |
|-------|-----------|---------|----------|
| `quantization` | `--quantization` | null | Critical — changes precision entirely |
| `load_format` | `--load-format` | "auto" | Critical — safetensors vs pytorch can differ |
| `kv_cache_dtype` | `--kv-cache-dtype` | "auto" | Critical — fp8/int8 KV cache changes attention |
| `max_num_batched_tokens` | `--max-num-batched-tokens` | null (inferred) | Potentially critical |
| `block_size` | `--block-size` | null (varies) | Potentially critical |
| `enable_prefix_caching` | `--enable-prefix-caching` | false | Potentially critical |
| `enable_chunked_prefill` | `--enable-chunked-prefill` | false | Potentially critical |
| `scheduling_policy` | `--scheduling-policy` | "fcfs" | Potentially critical |
| `disable_sliding_window` | `--disable-sliding-window` | false | Potentially critical |

All should be optional in the schema with sensible defaults. If present, passed to vLLM. If absent, vLLM uses its own defaults (which are pinned by the nix closure).

## 5. Parallelism fields

**Current:** `topology.mode` declares single_node/replicated/TP/PP but the server doesn't pass `--tensor-parallel-size` or `--pipeline-parallel-size` to vLLM. Topology is being removed (section 7).

**Change:** Add directly to `runtime.serving_engine`:
- `tensor_parallel_size` — integer, default 1, maps to `--tensor-parallel-size`
- `pipeline_parallel_size` — integer, default 1, maps to `--pipeline-parallel-size`
- `disable_custom_all_reduce` — boolean, default false, maps to `--disable-custom-all-reduce`

These are optional. If absent, defaults to single GPU. No indirection through topology mode — just declare the parallelism you want.

## 6. Replace nix_pin with container image digest

**Current:** `runtime.nix_pin` has `flake_ref` and `flake_hash`. Neither is consumed or verified. The flake provenance is not stored anywhere inside the built container, so there's no way to verify these from inside a running container.

**Change:**
- Remove `runtime.nix_pin` (flake_ref, flake_hash)
- Add `runtime.container_image_digest` — the OCI image sha256 digest (e.g. `sha256:a3cfd682bbd19d53cea020c0ba358a23b6618078e20004aa278be6d09e1c76ee`)
- This transitively pins everything the flake produced — vLLM version, torch version, CUDA toolkit, triton, all Python deps

**Verification:** At startup, the server reads its own container image digest:
```python
# The container ID is available from within the container:
# /proc/self/cgroup or HOSTNAME env var gives the container ID
# Then: docker inspect <id> --format '{{.Image}}' gives the image digest
```
Alternatively, bake the digest into the container at build time (write to `/etc/container-digest` during nix OCI build) and read it at runtime.

Compare against `runtime.container_image_digest` from the manifest. If mismatch, refuse to start (wrong container for this manifest).

**Why:** The container image digest is already computed by docker/nix, is cheap to verify, and covers the entire software closure. The flake ref/hash are build-time metadata that can't be verified at runtime without baking them in.

## 7. Explicit env var tracking

**Current:** `CUBLAS_WORKSPACE_CONFIG` and `PYTHONHASHSEED` are hardcoded in the server, not declared in the manifest.

**Change:** Add to `runtime.deterministic_knobs`:
- `cublas_workspace_config` — default `:4096:8`
- `pythonhashseed` — default `0`

These are already being set, just not tracked in the manifest. Making them explicit means the manifest fully declares the deterministic environment.

## 8. PCI IDs and topology

**Current:** `hardware_profile.gpu.pci_ids` and `hardware_profile.nic.pci_id` are required. `hardware_profile.topology` has `mode`, `node_count`, `rack_count`, `collective_fabric`.

**Change:**
- Make PCI IDs optional. They differ per machine and don't affect computation. Luke said: "for us we can just delete this because we can't guarantee that we have access to accelerators with the same PCI ID."
- Remove `gpu.vendor`. It's `const "nvidia"` — always nvidia, adds no information. The vendor is already apparent from `gpu.model` (e.g. "NVIDIA GH200 480GB").
- Remove all topology fields (`mode`, `node_count`, `rack_count`, `collective_fabric`). Luke said: "I think it's fine to not verify the topology. Because I just don't think it's gonna matter at the single node scale." These are stubs that aren't verified and add schema complexity. Parallelism is handled by the `--tensor-parallel-size` / `--pipeline-parallel-size` flags which should be in `serving_engine` directly (see section 5).
- Remove the schema conditional that requires `deterministic_dispatcher` when topology is non-single-node.
- Remove the schema conditional that requires `collective_stack` artifact when topology is TP/PP.
- Verify `gpu.driver_version` against `nvidia-smi --query-gpu=driver_version`. This is the host kernel driver — different versions can produce different cuBLAS results. Our previous cross-server experiment failed because of CUDA 12.8 vs 12.9.
- Verify `gpu.cuda_driver_version` against `torch.version.cuda` inside the container. This is the CUDA runtime from the nix closure — should always match if using the same container, but worth confirming.

## 9. Merge required_files into artifact_inputs

**Current:** Model files appear in two places:
- `model.required_files[]` — per-file with `role`, `path`, `digest`, `size_bytes`
- `artifact_inputs[]` — same files with `artifact_type`, `source_uri`, `immutable_ref`, `expected_digest`

This is redundant. The same weight shard appears in both lists with slightly different field names.

**Change:** Delete `model.required_files`. Add `path` and `role` fields to model-type entries in `artifact_inputs`:
```json
{
  "artifact_id": "hf-weights_shard-model-00001-of-00002.safetensors",
  "artifact_type": "model_weights",
  "role": "weights_shard",
  "path": "model-00001-of-00002.safetensors",
  "source_kind": "hf",
  "source_uri": "hf://Qwen/Qwen3-1.7B/model-00001-of-00002.safetensors",
  "immutable_ref": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
  "expected_digest": "sha256:abc123...",
  "size_bytes": 3456789
}
```

`artifact_inputs` becomes the single source of truth for all software — model files, CUDA toolkit, serving stack, everything. The server uses `artifact_type` to find model files for pre-serve hash verification (section 2).

**Update resolver:** `resolve_hf_model` already produces both `required_files` and `model_artifacts`. Change it to produce only `model_artifacts` with the `path` and `role` fields added. Delete `required_files` from `ResolvedHF` dataclass.

## 10. NIC verification

**Current:** `hardware_profile.nic` fields (firmware, link_speed_gbps, offloads) are validated by schema only. Never checked against the actual NIC.

**Actual values are fetchable** via `ethtool`:
```
driver: mlx5_core
firmware-version: 32.42.1000 (MT_0000000884)
Speed: 100000Mb/s
tcp-segmentation-offload: on
generic-segmentation-offload: on
rx-checksumming: on / tx-checksumming: on
rx-vlan-offload: on
```

**Change:** At manifest apply time, if NIC fields are present:
1. Identify the NIC device from `nic.model` (match driver name via `ethtool -i`)
2. Verify `nic.firmware` matches `ethtool -i <dev> | grep firmware-version`
3. Verify `nic.link_speed_gbps` matches `ethtool <dev> | grep Speed`
4. Verify offload states match `ethtool -k <dev>`:
   - `tso` → `tcp-segmentation-offload`
   - `gso` → `generic-segmentation-offload`
   - `checksum` → `rx-checksumming` / `tx-checksumming`
   - `vlan_strip` → `rx-vlan-offload`
5. If `strict_hardware=true`, mismatch is fatal. Otherwise, warn.

**Why:** NIC firmware and offload settings can affect network-level determinism. These values are stable per machine and cheap to query. Luke said the NIC fields are "the kind of thing that we would want to pin in a real setting."

**Implementation:** Add `_enforce_nic()` alongside existing `_enforce_hardware()` in `cmd/server/main.py`. Shell out to `ethtool` (already available on Lambda instances).

## 11. Prompts

**Current:** Full prompt text is inline in `requests[]`.

**Change:** No change for now. Luke said: "if all the experiments just have a small number of prompts, then it's fine to just leave the prompts in the manifest."

Future: For large prompt sets, support a `prompt_digest` + external file reference.

## 12. Remove activations

**Current:** Fake activation data in the runner (deterministic hash of token IDs, not real activations). `comparison.activations` comparator in the schema. Verifier compares placeholder values.

**Change:**
- Delete fake activation generation from `cmd/runner/vllm_runner.py` (line 115: `activations = [round(float((tok * 3) % 991) / 991.0, 8) for tok in tokens]`)
- Remove `activations` field from run bundle output
- Remove `comparison.activations` from the manifest schema
- Remove activation comparison from verifier

**Why:** Luke said: "don't worry about that. In the future we want to add activation logging, but it doesn't seem important right now." vLLM v1 doesn't expose activations through its API. Keeping fake data is misleading — it suggests verification that isn't happening.

## 13. Remove network section

**Current:** `network` has 11 required fields — `security_mode`, `egress_reproducibility`, `mtu`, `mss`, `tso`, `gso`, `checksum_offload`, `queue_mapping`, `ring_sizes`, `thread_affinity`, `internal_batching`. Also `comparison.network_egress` comparator and `runtime.allow_non_reproducible_egress`.

None of these are enforced. There's no userspace networking stack running. The fields are scaffolding from the network determinism design that was never wired to real infrastructure.

**Change:** Remove entirely:
- Delete `network` from the manifest schema
- Delete `comparison.network_egress` from the schema
- Delete `runtime.allow_non_reproducible_egress` from the schema
- Remove the schema conditional tying `security_mode=tls` to `allow_non_reproducible_egress`
- Remove `network.security_mode` from runner bundle metadata (`cmd/runner/main.py:554`)
- Remove NIC section from `hardware_profile` (firmware, link_speed, offloads — section 10 becomes moot)
- Remove NIC conformance check from runner (`cmd/runner/main.py:420`)

**Why:** No networking stack exists. These fields add schema complexity and required-field burden on every manifest without providing any verification or functionality. If network determinism is added in the future, the schema can be extended then.

**Impact on section 10 (NIC verification):** Section 10 planned to verify NIC fields via ethtool. With the network section removed, the NIC fields go away too. NIC verification is no longer needed.

## 14. Remove deterministic_dispatcher

**Current:** `deterministic_dispatcher` has `enabled`, `algorithm`, `request_order_source`, `replay_log_required`. Only `algorithm` is read — by `cmd/runner/dispatcher.py` and `cmd/coordinator/main.py` for multi-replica request routing.

**Change:** Remove from schema and code:
- Delete `deterministic_dispatcher` from the manifest schema
- Remove the schema conditional that required it for non-single-node topologies (already going away with topology removal in section 8)
- Keep `cmd/runner/dispatcher.py` and `cmd/coordinator/main.py` as standalone tools — they can accept dispatch config as CLI args if multi-replica is needed later, rather than coupling it into every manifest

**Why:** Single-node only for the demo. The dispatcher is dead code in our setup. Topology is being removed (section 8), which was the only trigger for requiring this section.

## 15. Remove batch_cardinality, batch_policy, and engine_trace

**Current:**
- `batch_cardinality` (`target_batch_size`, `min_requests`, `max_requests`) — not passed to vLLM. vLLM manages batching internally via continuous batching. The runner reads `target_batch_size` but only writes it as metadata into the bundle — it doesn't control actual batch sizes.
- `batch_policy` (`fixed`/`queued_fixed`) — recorded in the run bundle as metadata. Nothing enforces it.
- `engine_trace` (`enabled`, `events[]`) — controls which **fabricated** engine events are written to the bundle. The events are hardcoded: `attention_backend` is always `"flash_attention_2"`, reorder events always say no reordering happened, batch sizes are always `target_batch_size`. None of this reflects actual vLLM behavior.

**Change:** Remove all three from the schema and all code paths:
- Delete `runtime.batch_cardinality` from schema
- Delete `runtime.batch_policy` from schema
- Delete `runtime.engine_trace` from schema
- Remove synthetic engine event generation from `cmd/runner/main.py` and `cmd/runner/vllm_runner.py`
- Remove `target_batch_size` / `batch_policy` metadata from `cmd/capture/main.py`
- Remove engine trace from run bundle output

**Why:** These fields suggest observability and control that doesn't exist. vLLM's actual batching, scheduling, and engine decisions are not exposed through its API. Keeping fabricated data is misleading. Real engine observability would require instrumenting vLLM internals, which is future work.

**Note:** `max_num_seqs` (actual vLLM concurrency limit) is already in `serving_engine` and is passed to vLLM. `scheduling_policy` is being added in section 4. These cover the real levers for controlling vLLM's scheduling behavior.

## Implementation order

1. Schema changes (1, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15) — update `manifest.v1.schema.json`: collapse revisions, remove `remote_code` block, add serving_engine fields, replace `nix_pin` with `container_image_digest`, add deterministic_knobs env vars, remove `gpu.vendor`/topology/PCI IDs, remove `required_files`, add `role`/`path` to artifact_inputs, remove `comparison.activations`, remove network + `comparison.network_egress` + `allow_non_reproducible_egress`, remove `deterministic_dispatcher`, remove `batch_cardinality`/`batch_policy`/`engine_trace`
2. Resolver update (1, 3, 9) — update `cmd/resolver/main.py` and `pkg/common/hf_resolution.py`: use `weights_revision`, remove `required_files` and `remote_code` from `ResolvedHF`, produce enriched `artifact_inputs` with `path`/`role`
3. Server enforcement (2, 4, 5, 6, 8, 10) — update `cmd/server/main.py`: pass new serving_engine fields to vLLM, hash model files via artifact_inputs before serving, verify remote code digest via artifact_inputs, pass TP/PP flags, verify container image digest, verify driver versions, verify NIC via ethtool
4. Runner cleanup (12, 13) — remove fake activations from `cmd/runner/vllm_runner.py`, remove synthetic engine events from runner/capture, remove activation comparison from verifier
5. Tests — update unit and integration tests
6. Update manifest files — `manifests/qwen3-1.7b.manifest.json`
