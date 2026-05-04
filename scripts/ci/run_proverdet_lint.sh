#!/usr/bin/env bash
set -euo pipefail
# Lint the prover-verifier demo new code only. Skips silently if ruff is
# missing (the rest of the test suite shouldn't fail because a developer
# hasn't installed the dev tool yet).

CFG=experiments/prover-verifier-demo/ruff.toml
PATHS=(pkg/proverdet cmd/prover cmd/verifier_server cmd/verifier_cli)
TEST_GLOBS=()
for pattern in \
    'tests/unit/test_proverdet_*.py' \
    'tests/integration/test_prover_*.py' \
    'tests/integration/test_verifier_*.py' \
    'tests/e2e/test_prover_verifier_*.py'; do
    while IFS= read -r -d '' f; do
        TEST_GLOBS+=("$f")
    done < <(find . -path "./.git" -prune -o -path "./.venv" -prune -o -type f -name "${pattern##*/}" -print0 2>/dev/null \
        | grep -z "${pattern%/*}/" || true)
done

if ! command -v ruff >/dev/null 2>&1 && ! command -v uv >/dev/null 2>&1; then
    echo "[lint-proverdet] ruff/uv not installed; skipping" >&2
    exit 0
fi

# Only check paths that exist (early in the project, some won't yet).
EXISTING=()
for p in "${PATHS[@]}"; do
    if [ -e "$p" ]; then
        EXISTING+=("$p")
    fi
done
if [ "${#TEST_GLOBS[@]}" -gt 0 ]; then
    EXISTING+=("${TEST_GLOBS[@]}")
fi

if [ "${#EXISTING[@]}" -eq 0 ]; then
    echo "[lint-proverdet] no proverdet code yet; nothing to lint"
    exit 0
fi

if command -v ruff >/dev/null 2>&1; then
    ruff check --config "$CFG" "${EXISTING[@]}"
    ruff format --check --config "$CFG" "${EXISTING[@]}"
else
    uv run ruff check --config "$CFG" "${EXISTING[@]}"
    uv run ruff format --check --config "$CFG" "${EXISTING[@]}"
fi
