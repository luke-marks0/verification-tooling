#!/usr/bin/env bash
# End-to-end determinism verification via the live server.
#
# Sends identical request batches twice, converts captures to run bundles,
# and runs the verifier to compare them.
#
# Usage: deploy/lambda/verify.sh [--host HOST] [--port PORT] [--requests N]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VIRTUAL_ENV:-/home/ubuntu/venv}"
HOST="127.0.0.1"
PORT=8000
NUM_REQUESTS=8
MAX_TOKENS=64
OUT_BASE="/home/ubuntu/verify-runs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --requests) NUM_REQUESTS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

source "${VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

VERIFY_ID="verify-$(date +%Y%m%d-%H%M%S)"
VERIFY_DIR="${OUT_BASE}/${VERIFY_ID}"
mkdir -p "${VERIFY_DIR}"

# Find the active server's run directory
SERVER_DIR=$(ls -dt /home/ubuntu/server-runs/serve-live* 2>/dev/null | head -1)
if [ -z "$SERVER_DIR" ] || [ ! -f "$SERVER_DIR/manifest.resolved.json" ]; then
    echo "ERROR: No active server found. Start the server first with deploy/lambda/serve.sh"
    exit 1
fi

MANIFEST="$SERVER_DIR/manifest.resolved.json"
LOCKFILE="$SERVER_DIR/lockfile.built.v1.json"

echo "=== Determinism Verification ==="
echo "Server:   http://${HOST}:${PORT}"
echo "Manifest: ${MANIFEST}"
echo "Requests: ${NUM_REQUESTS} x ${MAX_TOKENS} max tokens"
echo "Output:   ${VERIFY_DIR}"
echo ""

# Health check
curl -sf "http://${HOST}:${PORT}/health" >/dev/null || { echo "ERROR: Server not responding"; exit 1; }

# Generate request payloads
python3 - "$NUM_REQUESTS" "$MAX_TOKENS" > "${VERIFY_DIR}/requests.json" <<'PYEOF'
import json, sys
num_requests = int(sys.argv[1])
max_tokens = int(sys.argv[2])
prompts = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "Write a Python function that computes factorial.",
    "What is the speed of light?",
    "Describe the water cycle briefly.",
    "What is machine learning?",
    "How does a compiler work?",
    "Explain the theory of relativity.",
    "What is DNA?",
    "How do neural networks learn?",
    "What is the Fibonacci sequence?",
    "Describe photosynthesis.",
    "What are black holes?",
    "How does encryption work?",
    "What is CRISPR?",
    "Explain plate tectonics.",
][:num_requests]
requests = []
for p in prompts:
    requests.append({
        "model": "Qwen/Qwen3-1.7B",
        "messages": [{"role": "user", "content": p}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": 42
    })
print(json.dumps(requests))
PYEOF

# ---- Session A: send requests, snapshot capture log ----
echo "--- Session A: sending ${NUM_REQUESTS} requests ---"

# Record capture log position before session A
CAPTURE_FILE="$SERVER_DIR/capture.jsonl"
PRE_A_LINES=$(wc -l < "$CAPTURE_FILE" 2>/dev/null || echo 0)

python3 - "${VERIFY_DIR}/requests.json" "http://${HOST}:${PORT}/v1/chat/completions" <<'PYEOF'
import json, urllib.request, sys
requests = json.load(open(sys.argv[1]))
url = sys.argv[2]
for i, req in enumerate(requests):
    body = json.dumps(req).encode()
    r = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as resp:
        resp.read()
    print(f"  [{i+1}/{len(requests)}] done")
print("Session A complete")
PYEOF

POST_A_LINES=$(wc -l < "$CAPTURE_FILE")

# Extract session A entries
SESSION_A_DIR="${VERIFY_DIR}/session-a"
mkdir -p "$SESSION_A_DIR"
tail -n +$((PRE_A_LINES + 1)) "$CAPTURE_FILE" | head -n $((POST_A_LINES - PRE_A_LINES)) > "$SESSION_A_DIR/capture.jsonl"
cp "$SERVER_DIR/boot_record.json" "$SESSION_A_DIR/" 2>/dev/null || true

echo "  Captured $((POST_A_LINES - PRE_A_LINES)) entries"

# ---- Session B: send same requests again ----
echo "--- Session B: sending ${NUM_REQUESTS} requests (replay) ---"

PRE_B_LINES=$(wc -l < "$CAPTURE_FILE")

python3 - "${VERIFY_DIR}/requests.json" "http://${HOST}:${PORT}/v1/chat/completions" <<'PYEOF'
import json, urllib.request, sys
requests = json.load(open(sys.argv[1]))
url = sys.argv[2]
for i, req in enumerate(requests):
    body = json.dumps(req).encode()
    r = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as resp:
        resp.read()
    print(f"  [{i+1}/{len(requests)}] done")
print("Session B complete")
PYEOF

POST_B_LINES=$(wc -l < "$CAPTURE_FILE")

SESSION_B_DIR="${VERIFY_DIR}/session-b"
mkdir -p "$SESSION_B_DIR"
tail -n +$((PRE_B_LINES + 1)) "$CAPTURE_FILE" | head -n $((POST_B_LINES - PRE_B_LINES)) > "$SESSION_B_DIR/capture.jsonl"
cp "$SERVER_DIR/boot_record.json" "$SESSION_B_DIR/" 2>/dev/null || true

echo "  Captured $((POST_B_LINES - PRE_B_LINES)) entries"

# ---- Convert captures to run bundles ----
echo ""
echo "--- Converting captures to run bundles ---"

BUNDLE_A="${VERIFY_DIR}/bundle-a"
python3 "${REPO_ROOT}/modules/inference/capture/main.py" \
    --server-dir "$SESSION_A_DIR" \
    --manifest "$MANIFEST" \
    --lockfile "$LOCKFILE" \
    --out-dir "$BUNDLE_A" \
    --session-id "session-a"
echo "  Bundle A: $BUNDLE_A/run_bundle.v1.json"

BUNDLE_B="${VERIFY_DIR}/bundle-b"
python3 "${REPO_ROOT}/modules/inference/capture/main.py" \
    --server-dir "$SESSION_B_DIR" \
    --manifest "$MANIFEST" \
    --lockfile "$LOCKFILE" \
    --out-dir "$BUNDLE_B" \
    --session-id "session-b"
echo "  Bundle B: $BUNDLE_B/run_bundle.v1.json"

# ---- Verify ----
echo ""
echo "--- Running verifier ---"

REPORT="${VERIFY_DIR}/verify_report.json"
SUMMARY="${VERIFY_DIR}/verify_summary.txt"

python3 "${REPO_ROOT}/modules/attestation/verifier/main.py" \
    --baseline "$BUNDLE_A/run_bundle.v1.json" \
    --candidate "$BUNDLE_B/run_bundle.v1.json" \
    --report-out "$REPORT" \
    --summary-out "$SUMMARY"

echo ""
cat "$SUMMARY"
echo ""

# Extract status
STATUS=$(python3 -c "import json; print(json.load(open('$REPORT'))['status'])")
echo "=== Verification result: ${STATUS} ==="
echo "Report: ${REPORT}"
echo "Summary: ${SUMMARY}"

if [ "$STATUS" = "conformant" ]; then
    exit 0
else
    exit 1
fi
