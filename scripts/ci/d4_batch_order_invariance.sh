#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

# ============================================================================
# D4-BOI: Batch & Order Invariance Test
# ============================================================================
# Proves that inference outputs are identical regardless of:
#   1. Request ordering (shuffle)
#   2. Batch size (max_num_seqs)
#
# For each model:
#   Run A: 100 requests in original order, max_num_seqs=64
#   Run B: same 100 requests shuffled, max_num_seqs=16
#   Compare: outputs matched by request ID must be bitwise identical
# ============================================================================

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [ "$GPU_COUNT" -lt 4 ]; then
    log "SKIP: need >= 4 GPUs, found $GPU_COUNT"; exit 0
fi

export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_DEBUG=WARN

# Models to test (passed as args, or default to both)
MODELS="${1:-dense,moe}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-/tmp/d4-boi-results}"
mkdir -p "$RESULTS_DIR"

# --------------------------------------------------------------------------
# Generate 100 diverse prompts deterministically
# --------------------------------------------------------------------------
generate_manifests() {
    local base_manifest="$1"
    local model_tag="$2"
    local out_dir="$3"

    python3 - "$base_manifest" "$model_tag" "$out_dir" << 'PYEOF'
import json, hashlib, sys, random
from pathlib import Path

base = json.loads(Path(sys.argv[1]).read_text())
tag = sys.argv[2]
out_dir = Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)

# 100 diverse prompts — deterministic via seed
topics = [
    "Explain how {} works in detail:",
    "Write a technical overview of {}:",
    "Describe the history and evolution of {}:",
    "Compare and contrast {} with alternatives:",
    "What are the key challenges in {}?",
]
subjects = [
    "TCP/IP networking", "quantum computing", "neural network backpropagation",
    "RSA encryption", "garbage collection in programming languages",
    "distributed consensus algorithms", "compiler optimization passes",
    "GPU shader programming", "database indexing strategies",
    "operating system virtual memory", "HTTP/2 protocol multiplexing",
    "Transformer attention mechanisms", "elliptic curve cryptography",
    "Linux kernel scheduling", "WebAssembly runtime design",
    "RAID storage architectures", "DNS resolution process",
    "TLS 1.3 handshake protocol", "MapReduce computation model",
    "binary search tree balancing",
]

requests = []
for i in range(100):
    topic_idx = i % len(topics)
    subject_idx = i % len(subjects)
    prompt = topics[topic_idx].format(subjects[subject_idx])
    requests.append({
        "id": f"req-{i:03d}",
        "prompt": prompt,
        "max_new_tokens": 128,
        "temperature": 0,
    })

# Run A: original order, max_num_seqs=64
manifest_a = json.loads(json.dumps(base))
manifest_a["run_id"] = f"{tag}-boi-ordered"
manifest_a["requests"] = requests
manifest_a["runtime"]["serving_engine"]["max_num_seqs"] = 64
a_path = out_dir / "manifest_a.json"
a_path.write_text(json.dumps(manifest_a, sort_keys=True, separators=(",", ":")) + "\n")

# Run B: shuffled order, max_num_seqs=16
rng = random.Random(12345)
shuffled = list(requests)
rng.shuffle(shuffled)
manifest_b = json.loads(json.dumps(base))
manifest_b["run_id"] = f"{tag}-boi-shuffled"
manifest_b["requests"] = shuffled
manifest_b["runtime"]["serving_engine"]["max_num_seqs"] = 16
b_path = out_dir / "manifest_b.json"
b_path.write_text(json.dumps(manifest_b, sort_keys=True, separators=(",", ":")) + "\n")

print(f"Generated manifests for {tag}: {len(requests)} requests")
print(f"  A: ordered, max_num_seqs=64 -> {a_path}")
print(f"  B: shuffled, max_num_seqs=16 -> {b_path}")
PYEOF
}

# --------------------------------------------------------------------------
# Run inference pipeline (resolve -> build -> run)
# --------------------------------------------------------------------------
run_inference() {
    local manifest="$1"
    local out_dir="$2"
    local label="$3"

    local resolved="$out_dir/resolved.lock.json"
    local built="$out_dir/built.lock.json"
    local run_dir="$out_dir/run"

    log "  [$label] Resolving..."
    python3 modules/inference/resolver/main.py --manifest "$manifest" --lockfile-out "$resolved"
    log "  [$label] Building..."
    python3 modules/build/builder/main.py --lockfile "$resolved" --lockfile-out "$built"
    log "  [$label] Running vLLM inference (100 requests)..."
    python3 modules/inference/runner/main.py --manifest "$manifest" --lockfile "$built" \
        --out-dir "$run_dir" --mode vllm --replica-id replica-0
    log "  [$label] Done."
}

# --------------------------------------------------------------------------
# Compare outputs by request ID (order-independent)
# --------------------------------------------------------------------------
compare_by_id() {
    local run_a="$1"
    local run_b="$2"
    local model_tag="$3"
    local report_path="$4"

    python3 - "$run_a" "$run_b" "$model_tag" "$report_path" << 'PYEOF'
import json, sys
from pathlib import Path

run_a_dir = Path(sys.argv[1])
run_b_dir = Path(sys.argv[2])
model_tag = sys.argv[3]
report_path = Path(sys.argv[4])

tokens_a = json.loads((run_a_dir / "observables" / "tokens.json").read_text())
tokens_b = json.loads((run_b_dir / "observables" / "tokens.json").read_text())
logits_a = json.loads((run_a_dir / "observables" / "logits.json").read_text())
logits_b = json.loads((run_b_dir / "observables" / "logits.json").read_text())

# Index by request ID
a_tokens_by_id = {r["id"]: r["tokens"] for r in tokens_a}
b_tokens_by_id = {r["id"]: r["tokens"] for r in tokens_b}
a_logits_by_id = {r["id"]: r["logits"] for r in logits_a}
b_logits_by_id = {r["id"]: r["logits"] for r in logits_b}

assert set(a_tokens_by_id.keys()) == set(b_tokens_by_id.keys()), "Request ID sets differ"

matches = 0
mismatches = []
total_tokens = 0
for rid in sorted(a_tokens_by_id.keys()):
    ta = a_tokens_by_id[rid]
    tb = b_tokens_by_id[rid]
    total_tokens += len(ta)
    if ta == tb:
        matches += 1
    else:
        first_diff = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        mismatches.append({"id": rid, "first_diff_pos": first_diff, "len_a": len(ta), "len_b": len(tb)})

logit_mismatches = 0
for rid in sorted(a_logits_by_id.keys()):
    la = a_logits_by_id[rid]
    lb = b_logits_by_id.get(rid, [])
    if la != lb:
        logit_mismatches += 1

report = {
    "model": model_tag,
    "test": "batch_order_invariance",
    "total_requests": len(a_tokens_by_id),
    "token_matches": matches,
    "token_mismatches": len(mismatches),
    "logit_mismatches": logit_mismatches,
    "total_tokens_compared": total_tokens,
    "status": "PASS" if len(mismatches) == 0 else "FAIL",
    "mismatch_details": mismatches[:10],
}

report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2) + "\n")

if mismatches:
    print("FAIL: %d/%d requests have token mismatches" % (len(mismatches), len(a_tokens_by_id)))
    for m in mismatches[:5]:
        print("  %s: diverges at token %d (len A=%d, B=%d)" % (m["id"], m["first_diff_pos"], m["len_a"], m["len_b"]))
    sys.exit(1)
else:
    print("PASS: all %d requests match (%d total tokens)" % (matches, total_tokens))
    print("  Order: shuffled vs original — identical")
    print("  Batch: max_num_seqs=16 vs 64 — identical")
PYEOF
}

# --------------------------------------------------------------------------
# Main: run for each model
# --------------------------------------------------------------------------

cd "$REPO_ROOT"

# Manifest paths — configurable via env vars for different GPU setups
DENSE_MANIFEST="${DENSE_MANIFEST:-modules/inference/manifests/qwen2.5-32b-tp4.manifest.json}"
MOE_MANIFEST="${MOE_MANIFEST:-modules/inference/manifests/qwen3-30b-moe-tp4.manifest.json}"
DENSE_TAG="${DENSE_TAG:-$(basename "$DENSE_MANIFEST" .manifest.json)}"
MOE_TAG="${MOE_TAG:-$(basename "$MOE_MANIFEST" .manifest.json)}"

if echo "$MODELS" | grep -q "dense"; then
    log "=== D4-BOI: $DENSE_TAG (dense, TP=4) ==="
    DENSE_DIR="$RESULTS_DIR/dense"
    generate_manifests "$DENSE_MANIFEST" "$DENSE_TAG" "$DENSE_DIR"

    run_inference "$DENSE_DIR/manifest_a.json" "$DENSE_DIR/a" "dense-A (ordered, batch=64)"
    run_inference "$DENSE_DIR/manifest_b.json" "$DENSE_DIR/b" "dense-B (shuffled, batch=16)"

    compare_by_id "$DENSE_DIR/a/run" "$DENSE_DIR/b/run" "$DENSE_TAG" "$RESULTS_DIR/report_dense.json"
    log "Dense model: DONE"
fi

if echo "$MODELS" | grep -q "moe"; then
    log "=== D4-BOI: $MOE_TAG (MoE, TP=4) ==="
    MOE_DIR="$RESULTS_DIR/moe"
    generate_manifests "$MOE_MANIFEST" "$MOE_TAG" "$MOE_DIR"

    run_inference "$MOE_DIR/manifest_a.json" "$MOE_DIR/a" "moe-A (ordered, batch=64)"
    run_inference "$MOE_DIR/manifest_b.json" "$MOE_DIR/b" "moe-B (shuffled, batch=16)"

    compare_by_id "$MOE_DIR/a/run" "$MOE_DIR/b/run" "$MOE_TAG" "$RESULTS_DIR/report_moe.json"
    log "MoE model: DONE"
fi

log "=== D4-BOI: All tests passed ==="
