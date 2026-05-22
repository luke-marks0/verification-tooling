#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "Running fast test suite (unit + integration + modules)"
bash scripts/ci/run_unittests.sh tests/unit
bash scripts/ci/run_unittests.sh tests/integration
bash scripts/ci/run_unittests.sh tests/modules
log "Fast test suite passed"
