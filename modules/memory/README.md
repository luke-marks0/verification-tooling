# memory — memory wipe + erasure attestation (PoSE)

**Purpose.** Cryptographically wipe a machine's memory (DRAM / GPU HBM / disk)
and *prove* the wipe happened — a Proof of Secure Erasure (PoSE-DB): fill the
region with verifiable noise, then answer challenge–response rounds that only a
holder of the noise can pass.

**Interface (today — runs on the target box).**

```bash
# On a GPU instance, as root (HBM access):
sudo .venv/bin/python3 scripts/benchmark.py --method crypto --disk-gb 500
```

The protocol pieces (`verifier`, `prover`, noise generators, DRAM/HBM/NVMe
backends) live in a self-contained `pose` package with its own tests.

**Artifacts.** Produces an erasure report (challenge success rate, timings).
Integrates with the attestation flow via `pkg/proverdet/erasure.py`
(`run_erasure`).

**Requirements.** Root on the target; a GPU for HBM wipes. Research-grade PoC.

**Underlying code.** `experiments/memory_wipe/src/pose/`
(protocol/prover/verifier/noise + `memory/{dram,hbm,nvme}.py`),
`experiments/memory_wipe/scripts/`.

**Status.** Facade present in `modules/memory/api.py`: `load_pose("protocol")`,
`load_pose("prover")`, `load_pose("memory.hbm")`, etc. (lazy — no GPU needed to
import). The `pose` package is **not relocated**: it's a separately-deployed
artifact (its own `pyproject.toml`, installed via `uv` on the target box), so the
facade imports it from `experiments/memory_wipe/src` rather than moving it (a
move would break the remote install workflow, which CI can't verify).
