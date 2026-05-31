#!/usr/bin/env bash
# Runs on the vast box. Brings up host_cluster -> recomp_cluster -> tap -> gateway
# sequentially. Each background process is launched via the
# `( nohup CMD </dev/null >LOG 2>&1 & )` subshell pattern so it is fully
# detached from this shell (and any parent ssh session): the subshell exits
# immediately after backgrounding, leaving the child reparented to init with
# SIGHUP ignored. `setsid` is unavailable on the Nix-only vast-test image.
#
# Sequential training note: training is single-threaded per cluster (see the
# train_lock in host_cluster.py and recomp_cluster.py). Both clusters are
# still running concurrently here — they just never train at the same time;
# Recomp only kicks off when the Tap's async /verify hits it.
#
# Required env: RUNNER_MODEL_PATH (set by launcher to the local HF snapshot dir).
set -euo pipefail

cd /root/dss
export PYTHONPATH=/root/dss
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export CUDA_VISIBLE_DEVICES=0
: "${RUNNER_MODEL_PATH:?must be set by launcher to local HF snapshot dir}"

LOG=/root/tap-train-logs
mkdir -p "$LOG"

# Bounded health wait. `curl` isn't on the Nix vast-test image. python3 stdlib
# `urllib` is, and is already required by the rest of the stack — use it as
# the probe. `i=$((i+1))` (not `(( i++ ))`) so the increment never returns 1
# under `set -e` on the first iteration (when i=0, `i++` evaluates to 0 which
# is arithmetic-false and trips set -e — silent abort, blank logs).
_http_200() {
    python3 -c '
import sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=3) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
' "$1"
}

wait_healthy() {
    local url="$1" label="$2" max_attempts="${3:-200}" sleep_s="${4:-3}"
    local i=0
    while [ "$i" -lt "$max_attempts" ]; do
        if _http_200 "$url"; then
            echo "[start] $label healthy"
            return 0
        fi
        sleep "$sleep_s"
        i=$((i+1))
    done
    echo "[start] $label NOT healthy after $((max_attempts * sleep_s))s; see $LOG" >&2
    return 1
}

# Host cluster: imports torch/transformers/peft eagerly on boot so /health
# only flips to 200 once those succeed. Adapter tarballs are saved under
# /tmp/host-cluster/adapters.
( nohup python3 demos/tap-train/servers/host_cluster.py \
    --port 8020 \
    --adapters-dir /tmp/host-cluster/adapters \
    </dev/null > "$LOG/host_cluster.out" 2>&1 & )
wait_healthy http://127.0.0.1:8020/health host_cluster 200 3

# Recomp cluster — same imports.
( nohup python3 demos/tap-train/servers/recomp_cluster.py \
    --port 8030 \
    --out-dir /tmp/recomp-cluster \
    </dev/null > "$LOG/recomp_cluster.out" 2>&1 & )
wait_healthy http://127.0.0.1:8030/health recomp_cluster 200 3

( nohup python3 demos/tap-train/servers/tap.py \
    --port 8010 --host-url http://127.0.0.1:8020 \
    --recomp-url http://127.0.0.1:8030 \
    </dev/null > "$LOG/tap.out" 2>&1 & )
wait_healthy http://127.0.0.1:8010/health tap 30 1

# Gateway is the only port exposed publicly on vast (mapped 8000:8000).
# Explicit --host 0.0.0.0 here makes the public bind a property of this
# script, not a default in gateway.py (which defaults to 127.0.0.1 to keep
# local runs and tests safe).
( nohup python3 demos/tap-train/servers/gateway.py \
    --port 8000 --host 0.0.0.0 --tap-url http://127.0.0.1:8010 \
    </dev/null > "$LOG/gateway.out" 2>&1 & )
wait_healthy http://127.0.0.1:8000/health gateway 30 1

echo "all four servers healthy"
