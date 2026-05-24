#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "Running schema gate"
require_cmd python3

python3 scripts/ci/schema_validate.py
python3 scripts/ci/schema_compat.py
python3 scripts/ci/check_canonical_json.py modules/core/schemas tests/fixtures/positive tests/fixtures/negative
python3 scripts/ci/check_conformance_catalog.py
python3 scripts/ci/fixture_validate.py
python3 scripts/ci/lockfile_validate.py --lockfile tests/fixtures/positive/lockfile.v1.example.json

log "Schema gate passed"
