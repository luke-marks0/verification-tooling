#!/usr/bin/env bash
# D6: Multi-node distributed determinism tests.
# Runs on the Ray head node. Requires a 4-node Ray cluster with 4 GPUs total.
#
# Tests:
#   1. PP=4 same-config determinism (Run A vs A')
#   2. PP=4 batch+order invariance (ordered/batch=64 vs shuffled/batch=16)
#   3. TP=4-over-TCP same-config determinism (stretch)
#
# Usage: ./d6_multinode_determinism.sh [--manifest MANIFEST] [--out-dir DIR]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MANIFEST="${MANIFEST:-$REPO_ROOT/modules/inference/manifests/qwen3-30b-moe-pp4-multinode.manifest.json}"
TP_MANIFEST="${TP_MANIFEST:-$REPO_ROOT/modules/inference/manifests/qwen3-30b-moe-tp4-multinode.manifest.json}"
LOCKFILE="${LOCKFILE:-$REPO_ROOT/modules/build/lockfiles/qwen3-30b-moe.lockfile.json}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/experiments/multinode-determinism/runs-$(date +%Y%m%d)}"
SKIP_TP="${SKIP_TP:-0}"

mkdir -p "$OUT_DIR"

# Verify Ray cluster
echo "=== Verifying Ray cluster ==="
ray status
GPU_COUNT=$(python3 -c "import ray; ray.init(); print(sum(n['Resources'].get('GPU', 0) for n in ray.nodes()))")
echo "Total GPUs in cluster: $GPU_COUNT"
if [ "$GPU_COUNT" -lt 4 ]; then
    echo "ERROR: Need 4 GPUs, found $GPU_COUNT"
    exit 1
fi

# Set NCCL determinism env for multi-node
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NET=Socket
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_BUFFSIZE=8388608
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG=WARN
export VLLM_MULTI_NODE=1
export VLLM_BATCH_INVARIANT=1
export RAY_ADDRESS=auto

echo "=== NCCL Settings ==="
env | grep ^NCCL | sort

# Helper: run inference and save results
run_inference() {
    local manifest="$1"
    local label="$2"
    local out_path="$OUT_DIR/$label"
    local extra_env="${3:-}"

    mkdir -p "$out_path"
    echo "--- Running: $label ---"

    env $extra_env python3 "$REPO_ROOT/modules/inference/runner/main.py" \
        --manifest "$manifest" \
        --lockfile "$LOCKFILE" \
        --out-dir "$out_path" \
        --mode vllm \
        --replica-id "replica-0" 2>&1 | tee "$out_path/runner.log"

    echo "  Output: $out_path"
}

# Helper: compare two runs by request ID
compare_runs() {
    local run_a="$1"
    local run_b="$2"
    local label="$3"

    echo ""
    echo "=== Comparing: $label ==="
    python3 << PYEOF
import json, glob, sys

def load_outputs(path):
    files = sorted(glob.glob(f"{path}/replica-*/observables.json"))
    if not files:
        files = sorted(glob.glob(f"{path}/observables.json"))
    if not files:
        print(f"  ERROR: no observables found in {path}")
        sys.exit(1)
    with open(files[0]) as f:
        data = json.load(f)
    return {r["id"]: r for r in data["request_outputs"]}

a = load_outputs("$run_a")
b = load_outputs("$run_b")

all_ids = sorted(set(a.keys()) | set(b.keys()))
match = 0
total = 0
total_tokens = 0
for rid in all_ids:
    if rid not in a or rid not in b:
        print(f"  {rid}: MISSING in {'A' if rid not in a else 'B'}")
        total += 1
        continue
    total += 1
    ra, rb = a[rid], b[rid]
    tokens_match = ra["tokens"] == rb["tokens"]
    total_tokens += len(ra["tokens"])
    if tokens_match:
        match += 1
    else:
        # Find first divergence point
        for i, (ta, tb) in enumerate(zip(ra["tokens"], rb["tokens"])):
            if ta != tb:
                print(f"  {rid}: DIVERGE at token {i} ({ta} vs {tb})")
                break
        else:
            print(f"  {rid}: DIVERGE (length {len(ra['tokens'])} vs {len(rb['tokens'])})")

status = "PASS" if match == total else "FAIL"
print(f"\n  {status}: {match}/{total} requests match ({total_tokens} total tokens)")

# Write summary
with open("$OUT_DIR/${label}_summary.json", "w") as f:
    json.dump({"test": "$label", "status": status, "match": match, "total": total, "total_tokens": total_tokens}, f, indent=2)

sys.exit(0 if match == total else 1)
PYEOF
}

# =====================================================
# Test 1: PP=4 Same-Config Determinism
# =====================================================
echo ""
echo "=========================================="
echo "Test 1: PP=4 Same-Config Determinism"
echo "=========================================="

run_inference "$MANIFEST" "pp4-run-a"
run_inference "$MANIFEST" "pp4-run-a-prime"
compare_runs "$OUT_DIR/pp4-run-a" "$OUT_DIR/pp4-run-a-prime" "pp4-same-config"

# =====================================================
# Test 2: PP=4 Batch + Order Invariance
# =====================================================
echo ""
echo "=========================================="
echo "Test 2: PP=4 Batch + Order Invariance"
echo "=========================================="

# Run A already done above (pp4-run-a with batch=64, ordered)
# Run B: shuffled order, batch=16
# Create a modified manifest with shuffled requests and smaller batch
python3 << 'PYEOF'
import json, random

with open("$MANIFEST") as f:
    m = json.load(f)

# Shuffle requests
random.seed(12345)
random.shuffle(m["requests"])

# Change batch size
m["runtime"]["serving_engine"]["max_num_seqs"] = 16
m["run_id"] = m["run_id"].replace("-001", "-shuffled-001")

out = "$OUT_DIR/pp4-manifest-shuffled.json"
with open(out, "w") as f:
    json.dump(m, f, indent=2)
print(f"Wrote shuffled manifest to {out}")
PYEOF

run_inference "$OUT_DIR/pp4-manifest-shuffled.json" "pp4-run-b-shuffled"
compare_runs "$OUT_DIR/pp4-run-a" "$OUT_DIR/pp4-run-b-shuffled" "pp4-batch-order-invariance"

# =====================================================
# Test 3 (Stretch): TP=4-over-TCP Same-Config
# =====================================================
if [ "$SKIP_TP" = "0" ]; then
    echo ""
    echo "=========================================="
    echo "Test 3: TP=4-over-TCP Same-Config (Stretch)"
    echo "=========================================="

    run_inference "$TP_MANIFEST" "tp4-tcp-run-a"
    run_inference "$TP_MANIFEST" "tp4-tcp-run-a-prime"
    compare_runs "$OUT_DIR/tp4-tcp-run-a" "$OUT_DIR/tp4-tcp-run-a-prime" "tp4-tcp-same-config"
else
    echo ""
    echo "=== Skipping TP=4-over-TCP test (SKIP_TP=1) ==="
fi

# =====================================================
# Summary
# =====================================================
echo ""
echo "=========================================="
echo "D6 Multi-Node Determinism — Summary"
echo "=========================================="
for f in "$OUT_DIR"/*_summary.json; do
    python3 -c "
import json
with open('$f') as fh:
    d = json.load(fh)
print(f\"  {d['test']}: {d['status']} ({d['match']}/{d['total']} requests, {d['total_tokens']} tokens)\")
"
done

echo ""
echo "Results in: $OUT_DIR"
