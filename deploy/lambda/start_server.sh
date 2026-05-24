#!/usr/bin/env bash
# Start the deterministic server on a remote Lambda node.
# Usage: deploy/lambda/start_server.sh <ip>
set -euo pipefail

IP="${1:?Usage: $0 <ip>}"

echo "=== Starting server on ${IP} ==="

ssh -o StrictHostKeyChecking=no "ubuntu@${IP}" bash -s <<'REMOTE'
set -euo pipefail
source /home/ubuntu/venv/bin/activate
export PYTHONPATH="/home/ubuntu/deterministic_serving_stack"
cd /home/ubuntu/deterministic_serving_stack

# Kill existing server
pkill -f "modules/inference/server/main.py" 2>/dev/null || true
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

RUN_DIR="/home/ubuntu/server-runs/serve-live"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

# Resolve + Build
python3 modules/inference/resolver/main.py \
    --manifest modules/inference/manifests/qwen3-1.7b.manifest.json \
    --lockfile-out "$RUN_DIR/lockfile.v1.json" \
    --manifest-out "$RUN_DIR/manifest.resolved.json" \
    --resolve-hf --hf-resolution-mode online 2>&1 | tail -2

# Build — use real Nix closure digest if Nix is available
BUILDER_ARGS="--lockfile $RUN_DIR/lockfile.v1.json --lockfile-out $RUN_DIR/lockfile.built.v1.json"
if command -v nix &>/dev/null; then
    CLOSURE_PATH=$(bash -l -c "nix build .#closure --print-out-paths --no-link 2>/dev/null" || true)
    if [ -n "$CLOSURE_PATH" ] && [ -d "$CLOSURE_PATH" ]; then
        CLOSURE_DIGEST=$(bash -l -c "nix path-info --json --recursive '$CLOSURE_PATH'" 2>/dev/null | python3 -c "
import json, sys, hashlib
data = json.load(sys.stdin)
entries = [{'path': k, **v} for k, v in sorted(data.items())] if isinstance(data, dict) else sorted(data, key=lambda x: x.get('path', ''))
print('sha256:' + hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
")
        echo "Nix closure digest: $CLOSURE_DIGEST"
        BUILDER_ARGS="$BUILDER_ARGS --builder-system nix --closure-digest $CLOSURE_DIGEST"
    else
        BUILDER_ARGS="$BUILDER_ARGS --builder-system equivalent"
    fi
else
    BUILDER_ARGS="$BUILDER_ARGS --builder-system equivalent"
fi
python3 modules/build/builder/main.py $BUILDER_ARGS 2>&1 | tail -2

# Start server in background
nohup python3 modules/inference/server/main.py \
    --manifest "$RUN_DIR/manifest.resolved.json" \
    --lockfile "$RUN_DIR/lockfile.built.v1.json" \
    --out-dir "$RUN_DIR" \
    --host 0.0.0.0 \
    --port 8000 \
    > "$RUN_DIR/server.log" 2>&1 &

# Wait for ready
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo "Server ready"
        exit 0
    fi
    sleep 3
done
echo "Server failed to start"
tail -20 "$RUN_DIR/server.log"
exit 1
REMOTE
