#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "D2: single-node runner determinism"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

MANIFEST="tests/fixtures/positive/manifest.v1.example.json"
NONSTRICT_MANIFEST="$TMP_DIR/manifest.nonstrict.json"
RESOLVED="$TMP_DIR/resolved.lock.json"
BUILT="$TMP_DIR/built.lock.json"
RUN_A="$TMP_DIR/run-a"
RUN_B="$TMP_DIR/run-b"
RUN_C="$TMP_DIR/run-c"
RUNTIME_HW="$TMP_DIR/runtime.hardware.json"
REPORT="$TMP_DIR/verify_report.json"
SUMMARY="$TMP_DIR/verify_summary.txt"

python3 modules/inference/resolver/main.py --manifest "$MANIFEST" --lockfile-out "$RESOLVED"
python3 modules/build/builder/main.py --lockfile "$RESOLVED" --lockfile-out "$BUILT"
python3 modules/inference/runner/main.py --manifest "$MANIFEST" --lockfile "$BUILT" --out-dir "$RUN_A" --replica-id replica-0
python3 modules/inference/runner/main.py --manifest "$MANIFEST" --lockfile "$BUILT" --out-dir "$RUN_B" --replica-id replica-0
python3 modules/attestation/verifier/main.py --baseline "$RUN_A/run_bundle.v1.json" --candidate "$RUN_B/run_bundle.v1.json" --report-out "$REPORT" --summary-out "$SUMMARY"

python3 - << 'PY' "$REPORT"
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report["status"] != "conformant":
    raise SystemExit(f"Expected conformant status, got: {report['status']}")
print("Verifier conformant status confirmed")
PY

python3 - << 'PY' "$MANIFEST" "$NONSTRICT_MANIFEST" "$RUNTIME_HW"
import copy
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
nonstrict = copy.deepcopy(manifest)
nonstrict["runtime"]["strict_hardware"] = False
Path(sys.argv[2]).write_text(json.dumps(nonstrict, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n", encoding="utf-8")

observed = copy.deepcopy(manifest["hardware_profile"])
observed["gpu"]["model"] = "H100-PCIe-80GB"
Path(sys.argv[3]).write_text(json.dumps(observed, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n", encoding="utf-8")
PY

if python3 modules/inference/runner/main.py --manifest "$MANIFEST" --lockfile "$BUILT" --out-dir "$TMP_DIR/run-strict-bad" --runtime-hardware "$RUNTIME_HW"; then
  echo "Expected strict_hardware=true run to fail on non-conforming hardware" >&2
  exit 1
fi

python3 modules/inference/resolver/main.py --manifest "$NONSTRICT_MANIFEST" --lockfile-out "$TMP_DIR/resolved.nonstrict.lock.json"
python3 modules/build/builder/main.py --lockfile "$TMP_DIR/resolved.nonstrict.lock.json" --lockfile-out "$TMP_DIR/built.nonstrict.lock.json"
python3 modules/inference/runner/main.py --manifest "$NONSTRICT_MANIFEST" --lockfile "$TMP_DIR/built.nonstrict.lock.json" --out-dir "$RUN_C" --runtime-hardware "$RUNTIME_HW"

python3 - << 'PY' "$RUN_C/run_bundle.v1.json"
import json
import sys
from pathlib import Path

bundle = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
hc = bundle["hardware_conformance"]
if hc["status"] != "non_conformant":
    raise SystemExit(f"Expected non_conformant hardware status, got {hc['status']}")
if hc["strict_hardware"] is not False:
    raise SystemExit("Expected strict_hardware=false in hardware conformance record")
if len(hc["diffs"]) == 0:
    raise SystemExit("Expected at least one hardware diff in non-strict mode")
print("Strict/non-strict hardware conformance behavior verified")
PY

python3 scripts/ci/mark_conformance.py --id SPEC-7.1-SINGLE-NODE-RUNNER
python3 scripts/ci/mark_conformance.py --id SPEC-7.3-STRICT-HARDWARE-REFUSE
python3 scripts/ci/mark_conformance.py --id SPEC-7.3-NONSTRICT-LABEL-NONCONFORMANT
python3 scripts/ci/mark_conformance.py --id SPEC-7.3-RUNTIME-VALIDATION-MUST

log "D2 passed"
