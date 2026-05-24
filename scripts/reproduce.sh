#!/usr/bin/env bash
# Reproduce the exact setup from a manifest.
#
# Given a manifest JSON, this script builds the Nix closure (or falls back
# to pip), downloads the model, resolves the lockfile, and starts the server.
#
# Usage:
#   scripts/reproduce.sh modules/inference/manifests/qwen3-1.7b.manifest.json
#   scripts/reproduce.sh modules/inference/manifests/qwen3-1.7b.manifest.json --run-tests
#
# With Nix: builds the exact closure from the pinned flake ref and verifies
#   the hash matches. This is the reproducible path.
#
# Without Nix: falls back to pip install of vLLM (best-effort, not hermetic).
set -euo pipefail

MANIFEST="${1:?Usage: $0 <manifest.json> [--run-tests]}"
RUN_TESTS=false
if [ "${2:-}" = "--run-tests" ]; then
    RUN_TESTS=true
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Reproduce from Manifest ==="
echo "Manifest: ${MANIFEST}"
echo ""

# ---- Step 1: Read pins from manifest ----
echo "--- Step 1: Reading manifest ---"
eval "$(python3 -c "
import json, sys
m = json.load(open('${MANIFEST}'))
nix = m.get('runtime', {}).get('nix_pin', {})
engine = m.get('runtime', {}).get('serving_engine', {})
model = m.get('model', {}).get('source', '')
if model.startswith('hf://'):
    model = model[5:]
print(f'FLAKE_REF={nix.get(\"flake_ref\", \"\")}')
print(f'FLAKE_HASH={nix.get(\"flake_hash\", \"\")}')
print(f'MAX_MODEL_LEN={engine.get(\"max_model_len\", 8192)}')
print(f'MAX_NUM_SEQS={engine.get(\"max_num_seqs\", 256)}')
print(f'GPU_MEM_UTIL={engine.get(\"gpu_memory_utilization\", 0.9)}')
print(f'DTYPE={engine.get(\"dtype\", \"auto\")}')
print(f'ATTENTION_BACKEND={engine.get(\"attention_backend\", \"FLASH_ATTN\")}')
print(f'MODEL_ID={model}')
")"

echo "  Flake ref: ${FLAKE_REF:-not set}"
echo "  Flake hash: ${FLAKE_HASH:-not set}"
echo "  Model: ${MODEL_ID}"
echo "  max_model_len: ${MAX_MODEL_LEN}"

# ---- Step 2: Build environment ----
echo ""
echo "--- Step 2: Build environment ---"

if command -v nix &>/dev/null && [ -n "$FLAKE_REF" ]; then
    echo "  Nix available — building closure from ${FLAKE_REF}"
    CLOSURE_PATH=$(bash -l -c "nix build '${FLAKE_REF}#closure' --print-out-paths --no-link 2>&1")
    echo "  Closure: ${CLOSURE_PATH}"

    # Verify hash
    ACTUAL_HASH=$(bash -l -c "nix path-info --json --recursive '${CLOSURE_PATH}'" 2>/dev/null | python3 -c "
import json, sys, hashlib
data = json.load(sys.stdin)
entries = [{'path': k, **v} for k, v in sorted(data.items())] if isinstance(data, dict) else sorted(data, key=lambda x: x.get('path', ''))
print('sha256:' + hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
")
    echo "  Actual hash:   ${ACTUAL_HASH}"
    echo "  Expected hash:  ${FLAKE_HASH}"

    if [ "$ACTUAL_HASH" = "$FLAKE_HASH" ]; then
        echo "  HASH MATCH: environment is identical"
    else
        echo "  WARNING: hash mismatch — environment may differ"
    fi

    # Use the Nix Python
    PYTHON="${CLOSURE_PATH}/bin/python3"
    if [ ! -x "$PYTHON" ]; then
        PYTHON="python3"
    fi
    USE_NIX=true
else
    echo "  Nix not available — falling back to pip"
    USE_NIX=false

    VENV="${REPO_ROOT}/.venv-reproduce"
    if [ ! -d "$VENV" ]; then
        python3 -m venv "$VENV"
    fi
    source "$VENV/bin/activate"
    pip install --upgrade pip setuptools wheel -q 2>&1 | tail -1

    # Best-effort: install vLLM (unpinned — this is NOT reproducible)
    pip install "vllm>=0.17" -q 2>&1 | tail -3
    # Fix torch CUDA if needed
    python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || \
        pip install torch --index-url https://download.pytorch.org/whl/cu128 -q 2>&1 | tail -2
    pip install jsonschema requests huggingface_hub pyyaml -q 2>&1 | tail -1

    PYTHON="python3"
fi

# ---- Step 3: Verify GPU ----
echo ""
echo "--- Step 3: Verify GPU ---"
$PYTHON -c "
import torch
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    print(f'  GPU: {torch.cuda.get_device_name()} (CC {cc[0]}.{cc[1]})')
    assert cc[0] >= 9, f'Batch invariance requires CC >= 9.0, got {cc[0]}.{cc[1]}'
else:
    print('  WARNING: No CUDA GPU')
"

# ---- Step 4: Download model ----
echo "--- Step 4: Downloading model ---"
$PYTHON -c "
from huggingface_hub import snapshot_download
p = snapshot_download('${MODEL_ID}')
print(f'  Cached at: {p}')
"

# ---- Step 5: Resolve + Build lockfile ----
echo "--- Step 5: Resolve and build ---"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RUN_DIR="${REPO_ROOT}/.reproduce-run"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

$PYTHON "${REPO_ROOT}/modules/inference/resolver/main.py" \
    --manifest "${MANIFEST}" \
    --lockfile-out "$RUN_DIR/lockfile.v1.json" \
    --manifest-out "$RUN_DIR/manifest.resolved.json" \
    --resolve-hf --hf-resolution-mode online

BUILDER_ARGS="--lockfile $RUN_DIR/lockfile.v1.json --lockfile-out $RUN_DIR/lockfile.built.v1.json"
if [ "$USE_NIX" = true ] && [ -n "$ACTUAL_HASH" ]; then
    BUILDER_ARGS="$BUILDER_ARGS --builder-system nix --closure-digest $ACTUAL_HASH"
else
    BUILDER_ARGS="$BUILDER_ARGS --builder-system equivalent"
fi
$PYTHON "${REPO_ROOT}/modules/build/builder/main.py" $BUILDER_ARGS

echo "  Resolved: $RUN_DIR/manifest.resolved.json"
echo "  Built:    $RUN_DIR/lockfile.built.v1.json"

# ---- Step 6: Start server ----
echo ""
echo "--- Step 6: Starting server ---"
export VLLM_BATCH_INVARIANT=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=0

$PYTHON "${REPO_ROOT}/modules/inference/server/main.py" \
    --manifest "$RUN_DIR/manifest.resolved.json" \
    --lockfile "$RUN_DIR/lockfile.built.v1.json" \
    --out-dir "$RUN_DIR/server" \
    --host 0.0.0.0 \
    --port 8000 &

SERVER_PID=$!

for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo ""
        echo "=== Server ready ==="
        echo "  PID: $SERVER_PID"
        echo "  URL: http://0.0.0.0:8000/v1/chat/completions"
        echo "  Nix: ${USE_NIX}"
        echo "  Hash verified: $([ \"$USE_NIX\" = true ] && [ \"$ACTUAL_HASH\" = \"$FLAKE_HASH\" ] && echo yes || echo no)"
        echo ""
        break
    fi
    sleep 3
done

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server failed to start"
    exit 1
fi

# ---- Step 7: Run tests (optional) ----
if [ "$RUN_TESTS" = true ]; then
    echo "--- Step 7: Running determinism tests ---"
    pip install pytest pytest-timeout -q 2>&1 | tail -1 || true

    export DETERMINISTIC_SERVER_URL="http://127.0.0.1:8000"

    $PYTHON -m pytest "${REPO_ROOT}/tests/determinism/test_network_determinism.py" -v --timeout=300 2>&1 || true
    echo ""
    export DETERMINISTIC_BATCH_SIZE=128
    $PYTHON -m pytest "${REPO_ROOT}/tests/determinism/test_large_batch_determinism.py" -v --timeout=600 2>&1 || true
fi

echo ""
echo "Server running (PID $SERVER_PID). Press Ctrl+C to stop."
wait $SERVER_PID
