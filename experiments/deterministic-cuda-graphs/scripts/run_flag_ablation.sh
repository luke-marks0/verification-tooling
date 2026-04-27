#!/usr/bin/env bash
set -euo pipefail
# Test which determinism flags are actually needed.
# Each config is run 3 times in separate processes and compared.

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")/data"
mkdir -p "$DATA_DIR"

run_config() {
    local config="$1"
    local n_runs=3

    echo ""
    echo "=============================================="
    echo "  Config: $config"
    echo "=============================================="

    for i in $(seq 0 $((n_runs - 1))); do
        echo ">>> Run $i"
        # Clear inherited env vars — the script sets its own
        env -u CUBLAS_WORKSPACE_CONFIG -u VLLM_BATCH_INVARIANT \
            PYTHONHASHSEED=0 \
            python3 "$SCRIPT_DIR/_flag_ablation.py" \
            --model "$MODEL" \
            --config "$config" \
            --run-id "$i" \
            --out "$DATA_DIR/ablation_${config}_${i}.json" 2>&1
    done

    echo ""
    echo "--- Comparing $config runs ---"
    python3 -c "
import json, sys

runs = []
for i in range(${n_runs}):
    with open('${DATA_DIR}/ablation_${config}_{}.json'.format(i)) as f:
        runs.append(json.load(f))

all_match = True
for i in range(1, ${n_runs}):
    mismatches = 0
    for j, (a, b) in enumerate(zip(runs[0], runs[i])):
        if a['content_hash'] != b['content_hash']:
            print(f'  MISMATCH prompt {j+1}: {a[\"prompt\"][:40]}...')
            mismatches += 1
            all_match = False
    print(f'  Run 0 vs Run {i}: {10 - mismatches}/10 match ({mismatches} mismatches)')

if all_match:
    print(f'  VERDICT [{\"$config\"}]: DETERMINISTIC')
else:
    print(f'  VERDICT [{\"$config\"}]: NON-DETERMINISTIC')
"
}

echo "=== Flag Ablation Study ==="
echo "Model: $MODEL"
echo "Testing which flags are needed for determinism with CUDA Graphs"

# Test each config
run_config "none"
run_config "cublas"
run_config "boi"
run_config "all"

echo ""
echo "=== Ablation complete ==="
