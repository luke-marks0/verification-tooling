#!/usr/bin/env bash
set -euo pipefail
# Run the deterministic CUDA graph experiment on a GPU machine.
# Usage: ssh ubuntu@<gpu-ip> 'bash -s' < run_on_gpu.sh

echo "=== Setting up deterministic CUDA graph experiment ==="

# Install uv if not present
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Clone the repo if needed
REPO_DIR="$HOME/deterministic_serving_stack"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/derpyplops/deterministic_serving_stack.git "$REPO_DIR"
else
    cd "$REPO_DIR" && git pull && cd -
fi

cd "$REPO_DIR"

SCRIPT_DIR="experiments/deterministic-cuda-graphs/scripts"
DATA_DIR="experiments/deterministic-cuda-graphs/data"
mkdir -p "$DATA_DIR"

echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Install vllm
echo "=== Installing vllm ==="
uv pip install --system vllm torch 2>/dev/null || pip install vllm torch

echo ""
echo "=== Running experiments ==="

# Control group: enforce_eager (known-good deterministic)
echo ""
echo ">>> Control: enforce_eager (c3 baseline)"
python3 "$SCRIPT_DIR/test_deterministic_graphs.py" \
    --approach eager \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --n-runs 3 \
    --out-dir "$DATA_DIR"

# Approach 1: Graphs + env vars only
echo ""
echo ">>> Approach 1: CUDA Graphs + determinism env vars"
python3 "$SCRIPT_DIR/test_deterministic_graphs.py" \
    --approach 1 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --n-runs 3 \
    --out-dir "$DATA_DIR"

# Approach 4: Same-process replay (tells us if replay is deterministic)
echo ""
echo ">>> Approach 4: Same-process replay test"
python3 "$SCRIPT_DIR/test_deterministic_graphs.py" \
    --approach 4 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --n-runs 5 \
    --out-dir "$DATA_DIR"

# Approach 2: + torch.use_deterministic_algorithms
echo ""
echo ">>> Approach 2: + torch.use_deterministic_algorithms(True)"
python3 "$SCRIPT_DIR/test_deterministic_graphs.py" \
    --approach 2 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --n-runs 3 \
    --out-dir "$DATA_DIR"

# Approach 3: + clock locking
echo ""
echo ">>> Approach 3: + GPU clock locking"
python3 "$SCRIPT_DIR/test_deterministic_graphs.py" \
    --approach 3 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --n-runs 3 \
    --out-dir "$DATA_DIR"

echo ""
echo "=== All experiments complete ==="
echo "Results in: $DATA_DIR/"
ls -la "$DATA_DIR/"
