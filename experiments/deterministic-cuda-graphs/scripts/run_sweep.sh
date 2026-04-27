#!/usr/bin/env bash
set -euo pipefail
# Full overhead sweep: 4 configs × 2 models × 5 batch sizes × 4 seq lengths = 160 runs

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")/data"
OUTFILE="$DATA_DIR/sweep.jsonl"
mkdir -p "$DATA_DIR"
: > "$OUTFILE"  # truncate

MODELS=("Qwen/Qwen2.5-1.5B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3")
CONFIGS=("baseline" "boi" "all" "eager")
BATCH_SIZES=(1 4 16 64 128)
SEQ_LENS=(16 128 512 2048)

TOTAL=$(( ${#MODELS[@]} * ${#CONFIGS[@]} * ${#BATCH_SIZES[@]} * ${#SEQ_LENS[@]} ))
COUNT=0

echo "=== Overhead Sweep (vLLM 0.19.1) ==="
echo "  Models: ${MODELS[*]}"
echo "  Configs: ${CONFIGS[*]}"
echo "  Batch sizes: ${BATCH_SIZES[*]}"
echo "  Seq lens: ${SEQ_LENS[*]}"
echo "  Total runs: $TOTAL"
echo "  Output: $OUTFILE"
echo ""

for model in "${MODELS[@]}"; do
    model_short="${model##*/}"
    for config in "${CONFIGS[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            for sl in "${SEQ_LENS[@]}"; do
                COUNT=$((COUNT + 1))
                echo "[$COUNT/$TOTAL] $model_short config=$config batch=$bs seq=$sl"

                TMPOUT="$DATA_DIR/_tmp_result.json"
                env -u CUBLAS_WORKSPACE_CONFIG -u VLLM_BATCH_INVARIANT \
                    PYTHONHASHSEED=0 \
                    python3 "$SCRIPT_DIR/_sweep_single.py" \
                    --model "$model" \
                    --config "$config" \
                    --batch-size "$bs" \
                    --max-tokens "$sl" \
                    > "$TMPOUT" 2>&1 || {
                    echo "  FAILED (exit $?), skipping"
                    cat "$TMPOUT" | tail -5
                    continue
                }

                # Extract the marked result line
                RESULT=$(grep 'RESULT_JSON:' "$TMPOUT" | tail -1 | sed 's/RESULT_JSON://')
                if [ -z "$RESULT" ]; then
                    echo "  FAILED: no JSON output"
                    tail -5 "$TMPOUT"
                    continue
                fi

                echo "$RESULT" >> "$OUTFILE"
                TOK=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['tok_per_s'])")
                echo "  -> $TOK tok/s"
            done
        done
    done
done

echo ""
echo "=== Sweep complete: $COUNT runs ==="
echo "Results: $OUTFILE"
wc -l "$OUTFILE"
