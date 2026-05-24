#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "D5: network egress determinism and divergence reporting"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

MANIFEST="tests/fixtures/positive/manifest.v1.example.json"
RESOLVED="$TMP_DIR/resolved.lock.json"
BUILT="$TMP_DIR/built.lock.json"
RUN1="$TMP_DIR/run-1"
RUN2="$TMP_DIR/run-2"
REPORT_OK="$TMP_DIR/verify_report_ok.json"
SUMMARY_OK="$TMP_DIR/verify_summary_ok.txt"
REPORT_BAD="$TMP_DIR/verify_report_bad.json"
SUMMARY_BAD="$TMP_DIR/verify_summary_bad.txt"

python3 modules/inference/resolver/main.py --manifest "$MANIFEST" --lockfile-out "$RESOLVED"
python3 modules/build/builder/main.py --lockfile "$RESOLVED" --lockfile-out "$BUILT"
python3 modules/inference/runner/main.py --manifest "$MANIFEST" --lockfile "$BUILT" --out-dir "$RUN1" --replica-id replica-0
python3 modules/inference/runner/main.py --manifest "$MANIFEST" --lockfile "$BUILT" --out-dir "$RUN2" --replica-id replica-0

python3 modules/attestation/verifier/main.py --baseline "$RUN1/run_bundle.v1.json" --candidate "$RUN2/run_bundle.v1.json" --report-out "$REPORT_OK" --summary-out "$SUMMARY_OK"

python3 - << 'PY' "$REPORT_OK"
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report["status"] != "conformant":
    raise SystemExit(f"Expected conformant report for identical runs, got {report['status']}")
print("Conformant report verified")
PY

python3 - << 'PY' "$RUN2/observables/network_egress.json" "$RUN2/run_bundle.v1.json"
import json
import sys
from pathlib import Path

from modules.core.common.deterministic import canonical_json_text, compute_bundle_digest, sha256_prefixed

network_path = Path(sys.argv[1])
bundle_path = Path(sys.argv[2])

network_data = json.loads(network_path.read_text(encoding="utf-8"))
network_data[0]["frame_hex"] = network_data[0]["frame_hex"][:-2] + "ff"
network_path.write_text(
    json.dumps(network_data, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
    encoding="utf-8",
)

bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
network_digest = sha256_prefixed(network_path.read_bytes())
bundle["observables"]["network_egress"]["digest"] = network_digest
bundle["network_provenance"]["capture_digest"] = network_digest
bundle["bundle_digest"] = compute_bundle_digest(bundle)
bundle_path.write_text(canonical_json_text(bundle), encoding="utf-8")
print("Mutated candidate network frame and updated bundle digests")
PY

python3 modules/attestation/verifier/main.py --baseline "$RUN1/run_bundle.v1.json" --candidate "$RUN2/run_bundle.v1.json" --report-out "$REPORT_BAD" --summary-out "$SUMMARY_BAD"

python3 - << 'PY' "$REPORT_BAD"
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report["status"] != "mismatch_outputs":
    raise SystemExit(f"Expected mismatch_outputs for network mutation, got {report['status']}")
for key in ("first_divergence", "numeric_diff_stats", "batch_trace_diffs", "network_trace_diffs"):
    if key not in report:
        raise SystemExit(f"Missing required mismatch field: {key}")
print("Mismatch report fields verified")
PY

python3 scripts/ci/mark_conformance.py --id SPEC-9.1-NETWORK-EGRESS
python3 scripts/ci/mark_conformance.py --id SPEC-9.1-NETWORK-USERSPACE-ROUTING
python3 scripts/ci/mark_conformance.py --id SPEC-9.3-CAPTURE-NONPERTURBING

log "D5 passed"
