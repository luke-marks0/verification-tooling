#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "D0: manifest -> lockfile determinism"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

MANIFEST="tests/fixtures/positive/manifest.v1.example.json"
LOCK1="$TMP_DIR/lock1.json"
LOCK2="$TMP_DIR/lock2.json"

python3 modules/inference/resolver/main.py --manifest "$MANIFEST" --lockfile-out "$LOCK1"
python3 modules/inference/resolver/main.py --manifest "$MANIFEST" --lockfile-out "$LOCK2"

cmp -s "$LOCK1" "$LOCK2"
python3 scripts/ci/lockfile_validate.py --lockfile "$LOCK1"
python3 scripts/ci/check_canonical_json.py "$LOCK1" "$LOCK2"
python3 scripts/ci/mark_conformance.py --id SPEC-5.1-LOCKFILE-DETERMINISM

log "D0 passed"
