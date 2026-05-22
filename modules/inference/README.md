# inference — deterministic inference (the c3 config)

**Purpose.** Bitwise-deterministic LLM inference: same weights, prompts, and
config flags → identical token outputs across independent servers.

**The "c3" config** (all three required):
1. `enforce_eager=True` — no CUDA Graphs / torch.compile *(declared in manifest `runtime`)*
2. `CUBLAS_WORKSPACE_CONFIG=:4096:8` — deterministic cuBLAS kernels
3. `VLLM_BATCH_INVARIANT=1` + `attention_backend=FLASH_ATTN` — batch-order invariance

Env vars MUST be set **before** `import torch`/`import vllm`. They're exposed as
`modules.inference.C3_ENV`.

**Interface.**

```python
from modules.inference import C3_ENV, run_inference, verify_runs

run_inference(manifest, lockfile, "/tmp/run-a", mode="synthetic")  # or mode="vllm"
report = verify_runs("/tmp/run-a/run_bundle.v1.json",
                     "/tmp/run-b/run_bundle.v1.json",
                     report_out="/tmp/report.json", summary_out="/tmp/summary.txt")
assert report["status"] == "conformant"
```

**Artifacts.** Consumes `manifest.v1` + `lockfile.v1`; produces `run_bundle.v1`
(tokens, logits, network egress) and `verify_report.v1`.

**Modes / requirements.**
- `synthetic` — no GPU; deterministic stub observables (CI, local dev).
- `vllm` — real inference; needs a GPU + vLLM. Serve long-running via `cmd/server`.

**Example.** `workflows/deterministic_inference_server.py`.

**Underlying code.** `cmd/runner` (batch runner), `cmd/server` (HTTP serving),
`pkg/manifest` (config model).
