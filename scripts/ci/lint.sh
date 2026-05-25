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
done < <(find scripts tests modules workflows -type f -name '*.py' | sort)

# Ruff lint over the product surface (config + ignores in pyproject.toml). Run via
# uv (pinned ruff from uv.lock) when available; skip gracefully otherwise so the
# script still runs on a bare checkout.
if command -v uv >/dev/null 2>&1; then
  uv run ruff check modules workflows scripts tests
elif command -v ruff >/dev/null 2>&1; then
  ruff check modules workflows scripts tests
else
  log "uv/ruff not available; skipping ruff lint"
fi

log "Lint checks passed"
