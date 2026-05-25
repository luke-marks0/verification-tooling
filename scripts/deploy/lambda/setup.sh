#!/usr/bin/env bash
# Setup script for Lambda Cloud H100 PCIe instance.
# Run this once after provisioning a 1x H100 PCIe instance on Lambda Cloud.
#
# Usage: ssh ubuntu@<lambda-ip> 'bash -s' < scripts/deploy/lambda/setup.sh
set -euo pipefail

echo "=== Deterministic Serving Stack: Lambda H100 PCIe Setup ==="

# Lambda instances come with CUDA and drivers pre-installed.
# Verify GPU availability.
nvidia-smi || { echo "ERROR: nvidia-smi failed. Is this a GPU instance?"; exit 1; }

echo "--- Installing system dependencies ---"
sudo apt-get update -qq
sudo apt-get install -y -qq git jq python3-venv python3-pip

echo "--- Creating Python venv ---"
python3 -m venv /home/ubuntu/venv
source /home/ubuntu/venv/bin/activate

echo "--- Installing vLLM + dependencies ---"
pip install --upgrade pip setuptools wheel
pip install "vllm>=0.8.0" jsonschema requests huggingface_hub

echo "--- Cloning repo ---"
if [ ! -d /home/ubuntu/deterministic_serving_stack ]; then
    git clone https://github.com/derpyplops/deterministic-serving-stack.git \
        /home/ubuntu/deterministic_serving_stack
fi

echo "--- Pre-downloading Qwen3-1.7B ---"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-1.7B', cache_dir='/home/ubuntu/.cache/huggingface')
print('Model downloaded successfully')
"

echo "--- Verifying batch invariance support ---"
python3 -c "
import vllm
print(f'vLLM version: {vllm.__version__}')
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    print(f'Compute capability: {cc[0]}.{cc[1]}')
    if cc[0] < 9:
        print('WARNING: Batch invariance requires compute capability >= 9.0')
    else:
        print('OK: GPU supports batch invariance')
"

echo ""
echo "=== Setup complete ==="
echo "Run the pipeline with: scripts/deploy/lambda/run.sh"
