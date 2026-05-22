#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "Running lint checks"
require_cmd rg
require_cmd bash
require_cmd python3

if rg -n "^(<<<<<<<|=======|>>>>>>>)" . --glob '!.git/*'; then
  printf 'Merge conflict markers found\n' >&2
  exit 1
fi

while IFS= read -r script; do
  bash -n "$script"
done < <(find scripts/ci -maxdepth 1 -type f -name '*.sh' | sort)

while IFS= read -r pyfile; do
  python3 -m py_compile "$pyfile"
done < <(find cmd pkg scripts tests modules workflows -type f -name '*.py' | sort)

log "Lint checks passed"
