#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "D1: runtime closure digest determinism"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

MANIFEST="tests/fixtures/positive/manifest.v1.example.json"
RESOLVED="$TMP_DIR/resolved.lock.json"
BUILT1="$TMP_DIR/built1.lock.json"
BUILT2="$TMP_DIR/built2.lock.json"

python3 modules/inference/resolver/main.py --manifest "$MANIFEST" --lockfile-out "$RESOLVED"
python3 modules/build/builder/main.py --lockfile "$RESOLVED" --lockfile-out "$BUILT1"
python3 modules/build/builder/main.py --lockfile "$RESOLVED" --lockfile-out "$BUILT2"

python3 - << 'PY' "$BUILT1" "$BUILT2"
import json
import sys
from pathlib import Path

left = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
right = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

if left["runtime_closure_digest"] != right["runtime_closure_digest"]:
    raise SystemExit("runtime_closure_digest differs across identical builds")
if left["canonicalization"]["lockfile_digest"] != right["canonicalization"]["lockfile_digest"]:
    raise SystemExit("lockfile_digest differs across identical builds")
if left["build"]["closure_inputs_digest"] != left["runtime_closure_digest"]:
    raise SystemExit("build.closure_inputs_digest must match runtime_closure_digest")

# The software stack is recorded via the Nix runtime closure
# (build.nix_closure), not per-component manifest artifacts. The D1 invariant
# is reproducibility: two identical builds must produce an identical build
# profile (closure metadata included).
if left["build"] != right["build"]:
    raise SystemExit("build profile differs across identical builds")

print("Builder determinism checks passed")
PY

python3 scripts/ci/lockfile_validate.py --lockfile "$BUILT1"
python3 scripts/ci/mark_conformance.py --id SPEC-6.1-RUNTIME-CLOSURE-DIGEST
python3 scripts/ci/mark_conformance.py --id SPEC-6.1-NIX-CLOSURE-CONTENT

log "D1 passed"
