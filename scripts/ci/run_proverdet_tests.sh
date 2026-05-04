#!/usr/bin/env bash
set -euo pipefail
# Run the proverdet unit tests. Exits 0 if no test files exist yet (early
# in the experiment) so that the pre-commit ritual still works.

shopt -s nullglob
matches=( tests/unit/test_proverdet_*.py )
if [ "${#matches[@]}" -eq 0 ]; then
    echo "[test-proverdet] no test_proverdet_*.py files yet; nothing to run"
    exit 0
fi

python3 -m unittest discover -s tests/unit -p 'test_proverdet_*.py' -v
