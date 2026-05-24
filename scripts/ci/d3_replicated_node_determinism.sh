#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "D3: replicated deterministic dispatch"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

MANIFEST="tests/fixtures/positive/manifest.v1.example.json"
DISPATCH1="$TMP_DIR/dispatch1.json"
DISPATCH2="$TMP_DIR/dispatch2.json"
REPLICAS="replica-0,replica-1,replica-2,replica-3"

python3 modules/inference/runner/dispatcher.py --manifest "$MANIFEST" --replicas "$REPLICAS" --out "$DISPATCH1"
python3 modules/inference/runner/dispatcher.py --manifest "$MANIFEST" --replicas "$REPLICAS" --out "$DISPATCH2"
cmp -s "$DISPATCH1" "$DISPATCH2"

RESOLVED="$TMP_DIR/resolved.lock.json"
BUILT="$TMP_DIR/built.lock.json"
RUN0="$TMP_DIR/run-0"
RUN1="$TMP_DIR/run-1"

python3 modules/inference/resolver/main.py --manifest "$MANIFEST" --lockfile-out "$RESOLVED"
python3 modules/build/builder/main.py --lockfile "$RESOLVED" --lockfile-out "$BUILT"
python3 modules/inference/runner/main.py --manifest "$MANIFEST" --lockfile "$BUILT" --out-dir "$RUN0" --replica-id replica-0
python3 modules/inference/runner/main.py --manifest "$MANIFEST" --lockfile "$BUILT" --out-dir "$RUN1" --replica-id replica-1

python3 - << 'PY' "$RUN0/observables/tokens.json" "$RUN1/observables/tokens.json"
import json
import sys
from pathlib import Path

left = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
right = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if left != right:
    raise SystemExit("Replicated token outputs diverged")
print("Replicated token outputs match")
PY

python3 scripts/ci/mark_conformance.py --id SPEC-10.1-REPLICATED-DISPATCH

log "D3 passed"
