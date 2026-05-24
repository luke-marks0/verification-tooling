#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "D4-TP: tensor-parallel runtime determinism (Qwen2.5-32B, TP=4)"

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [ "$GPU_COUNT" -lt 4 ]; then
    log "SKIP: need >= 4 GPUs, found $GPU_COUNT"
    exit 0
fi

log "Detected $GPU_COUNT GPUs"

# Pin NCCL collective algorithms for determinism
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_DEBUG=WARN

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

MANIFEST="modules/inference/manifests/qwen2.5-32b-tp4.manifest.json"
RESOLVED="$TMP_DIR/resolved.lock.json"
BUILT="$TMP_DIR/built.lock.json"
RUN_A="$TMP_DIR/run-a"
RUN_B="$TMP_DIR/run-b"
REPORT="$TMP_DIR/verify_report.json"
SUMMARY="$TMP_DIR/verify_summary.txt"

log "Resolving manifest..."
python3 modules/inference/resolver/main.py --manifest "$MANIFEST" --lockfile-out "$RESOLVED"

log "Building lockfile..."
python3 modules/build/builder/main.py --lockfile "$RESOLVED" --lockfile-out "$BUILT"

log "Run A: vLLM inference with TP=$GPU_COUNT..."
python3 modules/inference/runner/main.py --manifest "$MANIFEST" --lockfile "$BUILT" \
    --out-dir "$RUN_A" --mode vllm --replica-id replica-0

log "Run B: vLLM inference with TP=$GPU_COUNT..."
python3 modules/inference/runner/main.py --manifest "$MANIFEST" --lockfile "$BUILT" \
    --out-dir "$RUN_B" --mode vllm --replica-id replica-0

log "Verifying determinism..."
python3 modules/attestation/verifier/main.py \
    --baseline "$RUN_A/run_bundle.v1.json" \
    --candidate "$RUN_B/run_bundle.v1.json" \
    --report-out "$REPORT" \
    --summary-out "$SUMMARY"

python3 - << 'PY' "$REPORT"
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report["status"] != "conformant":
    print(f"FAIL: Expected conformant, got: {report['status']}")
    if "first_divergence" in report:
        print(f"  First divergence: {report['first_divergence']}")
    if "numeric_diff_stats" in report:
        print(f"  Numeric diff stats: {json.dumps(report['numeric_diff_stats'], indent=2)}")
    raise SystemExit(1)
print("TP determinism: CONFORMANT")
PY

python3 scripts/ci/mark_conformance.py --id SPEC-10.2-TP-RUNTIME-DETERMINISM
python3 scripts/ci/mark_conformance.py --id SPEC-10.2-COLLECTIVE-STACK-PIN

log "D4-TP passed"
