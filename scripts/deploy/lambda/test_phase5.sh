#!/usr/bin/env bash
# Phase 5 integration test: multi-node replicated serving.
#
# Requires two Lambda instances with the server running on each.
# Tests that the coordinator dispatches deterministically and
# cross-pod verification passes.
#
# Usage: scripts/deploy/lambda/test_phase5.sh <node1_ip> <node2_ip>
set -euo pipefail

NODE1="${1:-$(cat /tmp/lambda_instance_ip 2>/dev/null || echo '')}"
NODE2="${2:-$(cat /tmp/lambda_node2_ip 2>/dev/null || echo '')}"

if [ -z "$NODE1" ] || [ -z "$NODE2" ]; then
    echo "Usage: $0 <node1_ip> <node2_ip>"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

echo "=== Phase 5 Integration Test: Multi-Node Replicated Serving ==="
echo "Node 1: $NODE1"
echo "Node 2: $NODE2"
echo ""

# Step 1: Verify both nodes are healthy
echo "--- Step 1: Health check ---"
for NODE in "$NODE1" "$NODE2"; do
    if curl -sf "http://${NODE}:8000/health" >/dev/null 2>&1; then
        MODEL=$(curl -s "http://${NODE}:8000/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
        echo "  $NODE: OK (model: $MODEL)"
    else
        echo "  $NODE: DOWN — start the server first"
        exit 1
    fi
done

# Step 2: Test deterministic dispatch — send same request to both nodes independently
echo ""
echo "--- Step 2: Per-node determinism ---"

REQ='{"model":"Qwen/Qwen3-1.7B","messages":[{"role":"user","content":"What is the capital of France? One word."}],"temperature":0,"max_tokens":32,"seed":42}'

R1=$(curl -s "http://${NODE1}:8000/v1/chat/completions" -H "Content-Type: application/json" -d "$REQ")
R2=$(curl -s "http://${NODE2}:8000/v1/chat/completions" -H "Content-Type: application/json" -d "$REQ")

python3 - "$R1" "$R2" <<'PYEOF'
import json, sys
r1 = json.loads(sys.argv[1])
r2 = json.loads(sys.argv[2])
t1 = r1["choices"][0]["message"]["content"]
t2 = r2["choices"][0]["message"]["content"]
print(f"  Node 1: {t1[:80]}...")
print(f"  Node 2: {t2[:80]}...")
if t1 == t2:
    print("  CROSS-NODE DETERMINISM: PASS")
else:
    print("  CROSS-NODE DETERMINISM: FAIL")
    sys.exit(1)
PYEOF

# Step 3: Batch test — 16 requests to each node, compare outputs
echo ""
echo "--- Step 3: Batch cross-node comparison (16 requests) ---"

python3 - "$NODE1" "$NODE2" <<'PYEOF'
import json, urllib.request, sys

node1, node2 = sys.argv[1], sys.argv[2]

prompts = [
    "What is quantum computing?",
    "Explain photosynthesis.",
    "Write factorial in Python.",
    "What is the speed of light?",
    "Describe the water cycle.",
    "What is machine learning?",
    "How does a compiler work?",
    "Explain relativity.",
    "What is DNA?",
    "How do neural networks learn?",
    "What is Fibonacci?",
    "How does encryption work?",
    "What are black holes?",
    "Explain plate tectonics.",
    "How do batteries work?",
    "What is CRISPR?",
]

matches = 0
total = len(prompts)

for i, prompt in enumerate(prompts):
    req = json.dumps({
        "model": "Qwen/Qwen3-1.7B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 64,
        "seed": 42,
    }).encode()

    r1 = urllib.request.urlopen(
        urllib.request.Request(f"http://{node1}:8000/v1/chat/completions",
                              data=req, headers={"Content-Type": "application/json"}),
        timeout=120,
    )
    r2 = urllib.request.urlopen(
        urllib.request.Request(f"http://{node2}:8000/v1/chat/completions",
                              data=req, headers={"Content-Type": "application/json"}),
        timeout=120,
    )

    t1 = json.loads(r1.read())["choices"][0]["message"]["content"]
    t2 = json.loads(r2.read())["choices"][0]["message"]["content"]

    ok = t1 == t2
    if ok:
        matches += 1
    else:
        print(f"  [{i}] MISMATCH: {prompt[:40]}...")
        print(f"       Node1: {t1[:60]}...")
        print(f"       Node2: {t2[:60]}...")

print(f"\n  CROSS-NODE BATCH: {matches}/{total} identical")
if matches == total:
    print("  RESULT: PASS")
else:
    print("  RESULT: FAIL")
    sys.exit(1)
PYEOF

# Step 4: Coordinator simulation — route requests across both nodes
echo ""
echo "--- Step 4: Coordinator dispatch test ---"

python3 - "$NODE1" "$NODE2" <<'PYEOF'
import json, urllib.request, sys, hashlib

node1, node2 = sys.argv[1], sys.argv[2]
nodes = [f"http://{node1}:8000", f"http://{node2}:8000"]

prompts = [f"Question {i}: explain topic {i}" for i in range(8)]

# Dispatch using round_robin_hash (same as coordinator)
def dispatch_index(req_id, seq, n_replicas):
    seed = json.dumps({"id": req_id, "seq": seq}, sort_keys=True).encode()
    h = hashlib.sha256(seed).hexdigest()[:8]
    return int(h, 16) % n_replicas

# Run 1
results1 = []
for seq, prompt in enumerate(prompts):
    idx = dispatch_index(f"req-{seq}", seq, len(nodes))
    url = f"{nodes[idx]}/v1/chat/completions"
    req = json.dumps({
        "model": "Qwen/Qwen3-1.7B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 32, "seed": 42,
    }).encode()
    resp = urllib.request.urlopen(
        urllib.request.Request(url, data=req, headers={"Content-Type": "application/json"}),
        timeout=120,
    )
    text = json.loads(resp.read())["choices"][0]["message"]["content"]
    results1.append((idx, text))

# Run 2 (same dispatch, should hit same nodes)
results2 = []
for seq, prompt in enumerate(prompts):
    idx = dispatch_index(f"req-{seq}", seq, len(nodes))
    url = f"{nodes[idx]}/v1/chat/completions"
    req = json.dumps({
        "model": "Qwen/Qwen3-1.7B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 32, "seed": 42,
    }).encode()
    resp = urllib.request.urlopen(
        urllib.request.Request(url, data=req, headers={"Content-Type": "application/json"}),
        timeout=120,
    )
    text = json.loads(resp.read())["choices"][0]["message"]["content"]
    results2.append((idx, text))

# Compare
matches = sum(1 for (i1, t1), (i2, t2) in zip(results1, results2)
              if i1 == i2 and t1 == t2)
node_dist = {}
for idx, _ in results1:
    node_dist[idx] = node_dist.get(idx, 0) + 1

print(f"  Dispatch distribution: {node_dist}")
print(f"  Deterministic dispatch: {matches}/{len(prompts)} identical")
if matches == len(prompts):
    print("  RESULT: PASS")
else:
    print("  RESULT: FAIL")
    sys.exit(1)
PYEOF

echo ""
echo "=== Phase 5 Integration Test: COMPLETE ==="
