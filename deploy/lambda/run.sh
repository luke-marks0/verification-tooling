#!/usr/bin/env bash
# Run the full deterministic serving pipeline on a Lambda H100 PCIe instance.
#
# Usage:
#   deploy/lambda/run.sh [--runs N] [--manifest PATH]
#
# Runs the resolver -> builder -> runner pipeline, then optionally runs the
# verifier to compare repeated runs for batch invariance.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VIRTUAL_ENV:-/home/ubuntu/venv}"
MANIFEST="${REPO_ROOT}/modules/inference/manifests/qwen3-1.7b.manifest.json"
NUM_RUNS=2
OUT_BASE="/home/ubuntu/runs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) NUM_RUNS="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

source "${VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "=== Deterministic Serving Pipeline ==="
echo "Manifest: ${MANIFEST}"
echo "Runs: ${NUM_RUNS}"
echo ""

# Step 1: Resolve manifest -> lockfile (with real HF resolution)
echo "--- Step 1: Resolver ---"
LOCKFILE="${OUT_BASE}/lockfile.v1.json"
mkdir -p "${OUT_BASE}"

python3 "${REPO_ROOT}/modules/inference/resolver/main.py" \
    --manifest "${MANIFEST}" \
    --lockfile-out "${LOCKFILE}" \
    --resolve-hf \
    --hf-resolution-mode online

echo "Lockfile written to ${LOCKFILE}"
echo ""

# Step 2: Builder (reference descriptor, no Nix on Lambda)
echo "--- Step 2: Builder ---"
BUILT_LOCKFILE="${OUT_BASE}/lockfile.built.v1.json"

python3 "${REPO_ROOT}/modules/build/builder/main.py" \
    --lockfile "${LOCKFILE}" \
    --lockfile-out "${BUILT_LOCKFILE}" \
    --builder-system equivalent

echo "Built lockfile written to ${BUILT_LOCKFILE}"
echo ""

# Step 3: Runner (vLLM mode with batch invariance)
echo "--- Step 3: Runner (${NUM_RUNS} runs) ---"
BUNDLE_DIRS=()

for i in $(seq 1 "${NUM_RUNS}"); do
    RUN_DIR="${OUT_BASE}/run-${i}"
    echo "  Run ${i}/${NUM_RUNS} -> ${RUN_DIR}"

    python3 "${REPO_ROOT}/modules/inference/runner/main.py" \
        --manifest "${MANIFEST}" \
        --lockfile "${BUILT_LOCKFILE}" \
        --out-dir "${RUN_DIR}" \
        --mode vllm \
        --replica-id "replica-0"

    BUNDLE_DIRS+=("${RUN_DIR}")
    echo "  Run ${i} complete: ${RUN_DIR}/run_bundle.v1.json"
done
echo ""

# Step 4: Verify (compare run 1 vs run N for determinism)
if [ "${NUM_RUNS}" -ge 2 ]; then
    echo "--- Step 4: Verifier ---"
    BASELINE="${BUNDLE_DIRS[0]}/run_bundle.v1.json"
    REPORT_DIR="${OUT_BASE}/verify"
    mkdir -p "${REPORT_DIR}"

    for i in $(seq 2 "${NUM_RUNS}"); do
        CANDIDATE="${BUNDLE_DIRS[$((i-1))]}/run_bundle.v1.json"
        REPORT="${REPORT_DIR}/report-1-vs-${i}.json"
        SUMMARY="${REPORT_DIR}/summary-1-vs-${i}.txt"

        echo "  Comparing run 1 vs run ${i}..."
        python3 "${REPO_ROOT}/modules/attestation/verifier/main.py" \
            --baseline "${BASELINE}" \
            --candidate "${CANDIDATE}" \
            --report-out "${REPORT}" \
            --summary-out "${SUMMARY}"

        echo "  Report: ${REPORT}"
        if [ -f "${SUMMARY}" ]; then
            echo "  ---"
            cat "${SUMMARY}"
            echo "  ---"
        fi
    done
fi

echo ""
echo "=== Pipeline complete ==="
echo "Results in: ${OUT_BASE}"
echo ""
echo "Run bundles:"
for d in "${BUNDLE_DIRS[@]}"; do
    echo "  ${d}/run_bundle.v1.json"
done
if [ "${NUM_RUNS}" -ge 2 ]; then
    echo "Verification reports:"
    echo "  ${OUT_BASE}/verify/"
fi
