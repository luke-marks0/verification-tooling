#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "D4: tensor/pipeline trace determinism"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

MANIFEST_SRC="tests/fixtures/positive/manifest.v1.example.json"
MANIFEST_TP="$TMP_DIR/manifest.tp.json"

python3 - << 'PY' "$MANIFEST_SRC" "$MANIFEST_TP"
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest["run_id"] = "run-d4"
manifest["hardware_profile"]["topology"]["mode"] = "tensor_parallel"
manifest["hardware_profile"]["topology"]["node_count"] = 4
manifest["hardware_profile"]["topology"]["rack_count"] = 2
manifest["hardware_profile"]["topology"]["collective_fabric"] = "cross_rack"
manifest["deterministic_dispatcher"]["enabled"] = True
manifest["deterministic_dispatcher"]["algorithm"] = "sequence_map"
manifest["artifact_inputs"].append({
    "artifact_id": "collective-stack",
    "artifact_type": "collective_stack",
    "expected_digest": "sha256:" + ("c" * 64),
    "immutable_ref": "sha256:" + ("d" * 64),
    "name": "nccl-stack",
    "size_bytes": 512,
    "source_kind": "oci",
    "source_uri": "oci://registry.example/nccl@sha256:" + ("d" * 64),
})
Path(sys.argv[2]).write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n", encoding="utf-8")
print("Wrote TP manifest")
PY

RESOLVED="$TMP_DIR/resolved.lock.json"
BUILT="$TMP_DIR/built.lock.json"
RUN1="$TMP_DIR/run-1"
RUN2="$TMP_DIR/run-2"

python3 modules/inference/resolver/main.py --manifest "$MANIFEST_TP" --lockfile-out "$RESOLVED"
python3 modules/build/builder/main.py --lockfile "$RESOLVED" --lockfile-out "$BUILT"
python3 modules/inference/runner/main.py --manifest "$MANIFEST_TP" --lockfile "$BUILT" --out-dir "$RUN1" --replica-id replica-0
python3 modules/inference/runner/main.py --manifest "$MANIFEST_TP" --lockfile "$BUILT" --out-dir "$RUN2" --replica-id replica-0

python3 - << 'PY' "$RUN1/observables/engine_trace.json" "$RUN2/observables/engine_trace.json"
import json
import sys
from pathlib import Path

trace1 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
trace2 = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if trace1 != trace2:
    raise SystemExit("TP/PP engine trace diverged")
if not any(evt.get("event") == "collective_algorithm_selection" for evt in trace1):
    raise SystemExit("collective_algorithm_selection event missing")
print("TP/PP trace checks passed")
PY

python3 scripts/ci/mark_conformance.py --id SPEC-10.2-TPPP-TRACE
python3 scripts/ci/mark_conformance.py --id SPEC-10.2-COLLECTIVE-STACK-PIN

log "D4 passed"
