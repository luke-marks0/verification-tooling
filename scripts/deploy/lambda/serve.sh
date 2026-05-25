#!/usr/bin/env bash
# Start the deterministic serving stack on a Lambda instance.
#
# Usage:
#   scripts/deploy/lambda/serve.sh [--manifest PATH] [--port PORT]
#
# Prerequisites: run setup.sh first, or have vLLM + model cached.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
VENV="${VIRTUAL_ENV:-/home/ubuntu/venv}"
MANIFEST="${REPO_ROOT}/modules/inference/manifests/qwen3-1.7b.manifest.json"
OUT_BASE="/home/ubuntu/server-runs"
PORT=8000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

source "${VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RUN_ID="serve-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${OUT_BASE}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

echo "=== Deterministic Serving Stack ==="
echo "Manifest: ${MANIFEST}"
echo "Run dir:  ${RUN_DIR}"
echo ""

# Step 1: Resolve manifest -> lockfile + resolved manifest
echo "--- Step 1: Resolver ---"
LOCKFILE="${RUN_DIR}/lockfile.v1.json"
RESOLVED_MANIFEST="${RUN_DIR}/manifest.resolved.json"

python3 "${REPO_ROOT}/modules/inference/resolver/main.py" \
    --manifest "${MANIFEST}" \
    --lockfile-out "${LOCKFILE}" \
    --manifest-out "${RESOLVED_MANIFEST}" \
    --resolve-hf \
    --hf-resolution-mode online

echo "Lockfile: ${LOCKFILE}"
echo "Resolved manifest: ${RESOLVED_MANIFEST}"

# Step 2: Build — use real Nix closure digest if available
echo "--- Step 2: Builder ---"
BUILT_LOCKFILE="${RUN_DIR}/lockfile.built.v1.json"
BUILDER_ARGS=(
    --lockfile "${LOCKFILE}"
    --lockfile-out "${BUILT_LOCKFILE}"
)

if command -v nix &>/dev/null; then
    echo "  Nix detected — building real closure..."
    CLOSURE_PATH=$(bash -l -c "nix build '${REPO_ROOT}#closure' --print-out-paths --no-link 2>/dev/null" || true)
    if [ -n "$CLOSURE_PATH" ] && [ -d "$CLOSURE_PATH" ]; then
        CLOSURE_DIGEST=$(bash -l -c "nix path-info --json --recursive '$CLOSURE_PATH'" 2>/dev/null | python3 -c "
import json, sys, hashlib
data = json.load(sys.stdin)
if isinstance(data, dict):
    entries = [{'path': k, **v} for k, v in sorted(data.items())]
else:
    entries = sorted(data, key=lambda x: x.get('path', ''))
canonical = json.dumps(entries, sort_keys=True, separators=(',', ':'))
print('sha256:' + hashlib.sha256(canonical.encode()).hexdigest())
")
        echo "  Closure: ${CLOSURE_PATH}"
        echo "  runtime_closure_digest: ${CLOSURE_DIGEST}"
        BUILDER_ARGS+=(--builder-system nix --closure-digest "${CLOSURE_DIGEST}")
    else
        echo "  Nix build failed, falling back to equivalent"
        BUILDER_ARGS+=(--builder-system equivalent)
    fi
else
    echo "  Nix not available, using equivalent builder"
    BUILDER_ARGS+=(--builder-system equivalent)
fi

python3 "${REPO_ROOT}/modules/build/builder/main.py" "${BUILDER_ARGS[@]}"

echo "Built lockfile: ${BUILT_LOCKFILE}"

# Step 3: Start server
echo "--- Step 3: Starting server ---"
exec python3 "${REPO_ROOT}/modules/inference/server/main.py" \
    --manifest "${RESOLVED_MANIFEST}" \
    --lockfile "${BUILT_LOCKFILE}" \
    --out-dir "${RUN_DIR}" \
    --host 0.0.0.0 \
    --port "${PORT}"
