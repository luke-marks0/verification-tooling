#!/usr/bin/env bash
set -euo pipefail
# Run stress test: 100 prompts x 3 restarts, with and without CUDA Graphs

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")/data"
mkdir -p "$DATA_DIR"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export VLLM_BATCH_INVARIANT=1

echo "=== Stress Test: 100 prompts x 3 restarts ==="
echo ""

# Run with CUDA Graphs (no enforce_eager)
for i in 0 1 2; do
    echo ">>> Graphs run $i"
    python3 "$SCRIPT_DIR/_stress_test.py" --model "$MODEL" --run-id "$i" \
        --out "$DATA_DIR/stress_graphs_$i.json" 2>&1
    echo ""
done

# Compare
echo "=== Comparing graph runs ==="
python3 -c "
import json, sys

runs = []
for i in range(3):
    with open('$DATA_DIR/stress_graphs_{}.json'.format(i)) as f:
        runs.append(json.load(f))

all_match = True
for i in range(1, 3):
    mismatches = 0
    for j, (a, b) in enumerate(zip(runs[0], runs[i])):
        if a['content_hash'] != b['content_hash']:
            print(f'  MISMATCH prompt {j+1}: {a[\"prompt\"][:40]}...')
            print(f'    run0={a[\"content_hash\"][:24]}  run{i}={b[\"content_hash\"][:24]}')
            mismatches += 1
            all_match = False
    print(f'  Run 0 vs Run {i}: {100 - mismatches}/{100} match ({mismatches} mismatches)')

if all_match:
    print('  VERDICT: DETERMINISTIC (100/100 prompts x 3 restarts)')
else:
    print('  VERDICT: NON-DETERMINISTIC')
    sys.exit(1)
"

echo ""

# Run with enforce_eager for throughput comparison
echo ">>> Eager run (single, for throughput comparison)"
python3 "$SCRIPT_DIR/_stress_test.py" --model "$MODEL" --run-id 0 --eager \
    --out "$DATA_DIR/stress_eager_0.json" 2>&1

echo ""
echo "=== Done ==="
