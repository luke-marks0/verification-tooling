#!/usr/bin/env bash
set -euo pipefail
# Typecheck the prover-verifier demo new code only.

PROJECT=experiments/prover-verifier-demo/pyrightconfig.json

if ! command -v pyright >/dev/null 2>&1 && ! command -v uv >/dev/null 2>&1; then
    echo "[typecheck-proverdet] pyright/uv not installed; skipping" >&2
    exit 0
fi

# If none of the include paths exist yet, exit 0 (nothing to type-check).
HAS_TARGET=0
for p in modules/attestation/proverdet modules/attestation/prover modules/attestation/verifier_server modules/attestation/verifier_cli; do
    if [ -e "$p" ]; then
        HAS_TARGET=1
        break
    fi
done
if [ "$HAS_TARGET" -eq 0 ]; then
    echo "[typecheck-proverdet] no proverdet code yet; nothing to type-check"
    exit 0
fi

# Build the include list dynamically — pyright errors out (rather than warns)
# when a configured include directory does not exist, so we only pass the
# ones that exist on disk. If none exist, exit 0 (handled above).
INCLUDES=()
for p in modules/attestation/proverdet modules/attestation/prover modules/attestation/verifier_server modules/attestation/verifier_cli; do
    if [ -e "$p" ]; then
        INCLUDES+=("$p")
    fi
done

if command -v pyright >/dev/null 2>&1; then
    pyright --project "$PROJECT" "${INCLUDES[@]}"
else
    uv run pyright --project "$PROJECT" "${INCLUDES[@]}"
fi
