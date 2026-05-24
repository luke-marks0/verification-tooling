# attestation — correctness & integrity verification

**Purpose.** Prove the computation that produced an output was the claimed one,
without re-running everything: matmul attestation (Freivalds), token-level
commitments, and prover↔verifier replay verdicts (detect hidden
training/exfiltration in an inference workload).

**Interface (today).**

```bash
# Compare two run bundles -> verify_report.v1
python3 modules/attestation/verifier/main.py --baseline a/run_bundle.v1.json \
    --candidate b/run_bundle.v1.json --report-out report.json --summary-out summary.txt
```

```python
from modules.attestation.freivalds import execute_challenge, verify_response   # matmul attestation
from modules.attestation.e2e import commit_token, commit_token_stream          # token commitments
from modules.attestation.proverdet.verdict import replay_correctness, compute_budget, bandwidth_signal
```

**Artifacts.** Consumes `run_bundle.v1`; produces `verify_report.v1`,
`freivalds_attestation.v1`, `replay_evidence.v1`, `verifier_transcript_entry.v1`.

**Requirements.** CPU-only for verification (Freivalds is O(n²)); the prover side
runs on the serving GPU.

**Underlying code.** `modules/attestation/freivalds`, `modules/attestation/e2e`, `modules/attestation/proverdet`, `modules/attestation/verifier`,
`cmd/prover`, `modules/attestation/verifier_{cli,server}`. The prover-verifier-demo composes these
end-to-end (`experiments/prover-verifier-demo/`).

**Status.** Mature. Facade in `modules/attestation/api.py`: `attest_matmuls()`
(Freivalds round-trip, stdlib backend by default), `commit_token()` /
`commit_token_stream()`, `verify_runs()`, plus the `Challenge`/`MatmulSpec`/
`AttestationReport` types.
