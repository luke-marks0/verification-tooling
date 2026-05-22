# utils — provisioning & shared helpers

**Purpose.** The cross-cutting glue that doesn't belong to one capability:
cloud provisioning, the replay-server routine, and shared determinism helpers.

**Interface (commands & helpers).**

```bash
bash deploy/lambda/serve.sh --manifest manifests/qwen3-1.7b.manifest.json --port 8000
bash deploy/vast/setup_cluster.sh     # multi-node Ray cluster
python3 scripts/lambda_cli.py ...     # Lambda Cloud API wrapper
```

```python
from pkg.common.deterministic import canonical_json_bytes, sha256_prefixed
from pkg.common.contracts import validate_with_schema
```

**Artifacts.** N/A directly — provisions the environments the other capabilities
run in, and provides the canonical-JSON / digest helpers the whole spine relies on.

**Requirements.** Cloud API keys (Lambda / Vast) for provisioning; nothing for
the helpers.

**Underlying code.** `deploy/{lambda,vast,warden}/`, `scripts/lambda_cli.py`,
`demo/run_demo.py`, `pkg/common/`.

**Status.** Facade in `modules/utils/api.py` re-exports the canonical-JSON /
digest / schema-validation helpers (`canonical_json_bytes`, `sha256_prefixed`,
`validate_with_schema`, …). Provisioning remains rough shell scripts under
`deploy/` (functional for manual ops); a "replay-server" sub-routine is a later
follow-up.
