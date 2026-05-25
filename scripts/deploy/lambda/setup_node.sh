#!/usr/bin/env bash
# Setup a Lambda node for deterministic serving. Idempotent.
# Usage: scripts/deploy/lambda/setup_node.sh <ip>
set -euo pipefail

IP="${1:?Usage: $0 <ip>}"
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

echo "=== Setting up node ${IP} ==="

# Sync repo
echo "--- Syncing repo ---"
rsync -az --exclude '.git' --exclude '__pycache__' --exclude '.pytest_cache' \
    "${REPO_ROOT}/" "ubuntu@${IP}:/home/ubuntu/deterministic_serving_stack/"

# Remote setup
ssh -o StrictHostKeyChecking=no "ubuntu@${IP}" bash -s <<'REMOTE'
set -euo pipefail

python3 -m venv /home/ubuntu/venv 2>/dev/null || true
source /home/ubuntu/venv/bin/activate

pip install --upgrade pip setuptools wheel -q 2>&1 | tail -1
pip install "vllm>=0.8.0" jsonschema pytest-timeout -q 2>&1 | tail -3

# Fix torch CUDA
pip uninstall torch -y -q 2>/dev/null || true
pip install torch --index-url https://download.pytorch.org/whl/cu128 -q 2>&1 | tail -2

# Download model
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-1.7B')
print('Model cached')
"

# Verify
python3 -c "
import torch, vllm
print(f'vLLM {vllm.__version__}, PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"none\"}')
cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0,0)
print(f'CC: {cc[0]}.{cc[1]}')
"
REMOTE

echo "=== Node ${IP} ready ==="
