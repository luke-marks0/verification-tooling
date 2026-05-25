# deploy — running the stack on real GPU hardware

Operational scripts that take the repo from "code on disk" to "running on a real
GPU box." This is the only path that exercises **GPU bitwise determinism** and
computes the **real Nix closure pin** — GitHub CI is CPU-only and proves none of
that (see [`workflows/`](../workflows/) for the CPU recipes).

Three targets:

| Dir | Target | Use it for |
|---|---|---|
| `lambda/` | Lambda Cloud, single H100/GH200 | provision a box, serve, verify determinism end-to-end |
| `vast/` | vast.ai, 4× H100 SXM cluster | multi-node / cross-node determinism (the D6 gate) |
| `warden/` | any Linux host | run the active warden (network covert-channel scrubber) as a service |

## `lambda/` — single node

Prereqs: `LAMBDALABS_API_KEY` in the env and an SSH key named `macbook 2025`
registered on Lambda. Typical flow:

```bash
scripts/deploy/lambda/grab_instance.sh        # poll until a CC>=9.0 GPU is free, then launch it
scripts/deploy/lambda/setup_node.sh   <ip>    # install vLLM + deps on the box (idempotent)
scripts/deploy/lambda/start_server.sh <ip>    # resolve -> build (real nix closure if available) -> serve
scripts/deploy/lambda/verify.sh               # send identical batches twice, compare bundles -> conformant?
```

Other scripts:
- `setup.sh` — the one-shot install body (`setup_node.sh` runs it remotely; or `ssh ubuntu@<ip> 'bash -s' < scripts/deploy/lambda/setup.sh`).
- `serve.sh [--manifest PATH] [--port PORT]` — local variant of the resolve→build→serve flow (defaults: `qwen3-1.7b` manifest, port 8000).
- `run.sh [--runs N] [--manifest PATH]` — drive the resolver→builder→runner pipeline directly, then verify repeated runs.
- `run_vllm_bi_tests.sh <ip>` — vLLM batch-invariance tests on the instance.
- `test_phase5.sh <node1_ip> <node2_ip>` — two-node replicated-serving integration test.

`start_server.sh`/`serve.sh` build the **real** software pin when `nix` is on the
box (`nix build .#closure` → `--closure-digest`); without nix they fall back to
the metadata-only descriptor.

## `vast/` — multi-node cluster

```bash
scripts/deploy/vast/grab_cluster.sh                       # provision 4x H100 SXM -> writes cluster.env
scripts/deploy/vast/setup_cluster.sh <head> <w1> <w2> <w3>  # Ray cluster + multi-node vLLM (args are host:port)
scripts/deploy/vast/teardown_cluster.sh [contract_id ...]   # destroy nodes (no args = ALL instances — careful)
```

## `warden/` — active warden service

```bash
sudo scripts/deploy/warden/setup.sh install   # deps + config (NetfilterQueue, libnetfilter-queue)
sudo scripts/deploy/warden/setup.sh start     # add iptables NFQUEUE rules
sudo scripts/deploy/warden/setup.sh stop      # remove the rules
sudo scripts/deploy/warden/setup.sh status    # show rules + warden status
```

`warden.service` is the systemd unit for running it persistently.
