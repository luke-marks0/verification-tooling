#!/usr/bin/env bash
# Run vLLM batch invariance tests on a Lambda instance.
# Usage: scripts/deploy/lambda/run_vllm_bi_tests.sh <ip>
set -euo pipefail

IP="${1:-$(cat /tmp/lambda_instance_ip 2>/dev/null || echo '')}"
if [ -z "$IP" ]; then
    echo "Usage: $0 <lambda-instance-ip>"
    echo "   Or: write IP to /tmp/lambda_instance_ip"
    exit 1
fi

echo "=== Running vLLM Batch Invariance Tests on ${IP} ==="

ssh -o StrictHostKeyChecking=no "ubuntu@${IP}" bash -s <<'REMOTE_SCRIPT'
set -euo pipefail

echo "--- System info ---"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
echo ""

echo "--- Setting up venv ---"
python3 -m venv /home/ubuntu/venv 2>/dev/null || true
source /home/ubuntu/venv/bin/activate
pip install --upgrade pip setuptools wheel -q

echo "--- Installing vLLM ---"
pip install "vllm>=0.8.0" -q

echo "--- Installing test dependencies ---"
pip install pytest -q

echo "--- Cloning vLLM repo for test files ---"
if [ ! -d /home/ubuntu/vllm ]; then
    git clone --depth 1 https://github.com/vllm-project/vllm.git /home/ubuntu/vllm
else
    cd /home/ubuntu/vllm && git pull --ff-only || true
fi

echo "--- Pre-downloading Qwen3-1.7B ---"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-1.7B', cache_dir='/home/ubuntu/.cache/huggingface')
print('Model cached')
"

echo ""
echo "--- Verifying GPU compute capability ---"
python3 -c "
import torch
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f'GPU: {name}, Compute Capability: {cc[0]}.{cc[1]}')
    assert cc[0] >= 9, f'Need CC >= 9.0, got {cc[0]}.{cc[1]}'
    print('OK: Batch invariance supported')
else:
    print('ERROR: No CUDA GPU')
    exit 1
"

echo ""
echo "--- Running batch invariance tests ---"
export VLLM_BATCH_INVARIANT=1
export VLLM_TEST_MODEL="Qwen/Qwen3-1.7B"

cd /home/ubuntu/vllm
python3 -m pytest tests/v1/determinism/test_batch_invariance.py \
    -v \
    -x \
    --timeout=600 \
    -k "test_simple_generation or test_logprobs_bitwise_batch_invariance" \
    2>&1

echo ""
echo "=== Tests complete ==="
REMOTE_SCRIPT
