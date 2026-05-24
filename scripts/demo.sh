#!/usr/bin/env bash
# End-to-end e2e-audit demo for reviewers.
#
# Assumes: a CUDA H100 / GH200 node (compute capability ≥ 9.0) with
# nvidia driver + CUDA toolkit already installed (e.g. Lambda Cloud).
# Builds a venv, installs vllm from pip, starts the deterministic server
# with the audit-enabled smoke manifest, then runs the replay loop.
#
# Usage:
#   ./scripts/demo.sh          # run everything, tear down at the end
#   ./scripts/demo.sh --keep   # leave the server running after audit
set -uo pipefail

KEEP=false
[ "${1:-}" = "--keep" ] && KEEP=true

log(){ echo "=== [$(date +%H:%M:%S)] $* ===" >&2; }
fail(){ echo "FAIL: $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

log "host sanity"
uname -m
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || fail "no nvidia gpu visible"
python3 --version || fail "python3 missing"

VENV="$REPO_ROOT/.demo-venv"
RUN_DIR="$REPO_ROOT/.demo-run"

log "install python deps into $VENV"
rm -rf "$VENV"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip -q
# Install CUDA torch FIRST so pip doesn't later substitute a CPU wheel.
# Lambda GH200/H100 AMIs ship CUDA 12.8 driver; adjust cu128 if yours differs.
pip install -q --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 2>&1 | tail -3
pip install -q "vllm==0.17.1" pydantic jsonschema requests huggingface_hub pyyaml 2>&1 | tail -3
python3 -c "import vllm, torch; print(f'vllm={vllm.__version__} torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"—\"}')"

log "resolve manifest and build lockfile"
MANIFEST="$REPO_ROOT/experiments/e2e-audit/scripts/smoke.manifest.json"
rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

python3 modules/inference/resolver/main.py \
  --manifest "$MANIFEST" \
  --lockfile-out "$RUN_DIR/lockfile.v1.json" \
  --manifest-out "$RUN_DIR/manifest.resolved.json" \
  --resolve-hf --hf-resolution-mode online

python3 modules/build/builder/main.py \
  --lockfile "$RUN_DIR/lockfile.v1.json" \
  --lockfile-out "$RUN_DIR/lockfile.built.v1.json" \
  --builder-system equivalent

log "start deterministic server"
export VLLM_BATCH_INVARIANT=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=0

nohup python3 modules/inference/server/main.py \
  --manifest "$RUN_DIR/manifest.resolved.json" \
  --lockfile "$RUN_DIR/lockfile.built.v1.json" \
  --out-dir "$RUN_DIR/server" \
  --skip-boot-validation \
  --host 0.0.0.0 --port 8000 > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
disown
echo "server pid=$SERVER_PID log=$RUN_DIR/server.log"
if [ "$KEEP" = "false" ]; then
  trap "kill $SERVER_PID 2>/dev/null; wait 2>/dev/null" EXIT
fi

log "wait for vllm ready (up to 15 min)"
ready=0
for i in $(seq 1 180); do
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "--- server died; log tail: ---"; tail -200 "$RUN_DIR/server.log"
    fail "server died"
  fi
  code=$(curl -sf -o /tmp/m.json -w "%{http_code}" http://localhost:8000/manifest 2>/dev/null || echo "000")
  if [ "$code" = "200" ] && grep -q '"vllm_healthy": true' /tmp/m.json; then
    echo "ready after ${i}x5s"; ready=1; break
  fi
  sleep 5
done
[ "$ready" = "1" ] || { tail -200 "$RUN_DIR/server.log"; kill $SERVER_PID 2>/dev/null; fail "vllm not ready"; }

curl -s http://localhost:8000/manifest | python3 -c "
import json, sys
d = json.load(sys.stdin)
ac = d.get('active_config', {})
print('model:', ac.get('model'))
print('seed:', ac.get('seed'))
print('batch_invariance:', ac.get('batch_invariance'))
"

log "run audit replay"
python3 experiments/e2e-audit/scripts/vast_audit_replay.py --server http://localhost:8000
RC=$?
if [ "$KEEP" = "true" ]; then
  echo "server still running at http://localhost:8000 (pid=$SERVER_PID)"
fi
[ "$RC" = "0" ] && log "DEMO OK" || { log "AUDIT FAILED rc=$RC"; exit $RC; }
