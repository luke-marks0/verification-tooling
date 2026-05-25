#!/usr/bin/env bash
# Set up a Ray cluster across 4 vast.ai nodes for multi-node vLLM inference.
# Usage: ./setup_cluster.sh <head_ssh> <worker1_ssh> <worker2_ssh> <worker3_ssh>
#
# Each arg is "host:port" for SSH (e.g., ssh5.vast.ai:12345).
# Assumes all nodes are running and accessible via SSH as root.
set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 <head_host:port> <worker1_host:port> <worker2_host:port> <worker3_host:port>"
    exit 1
fi

HEAD_SSH="$1"; shift
WORKERS=("$@")

ssh_cmd() {
    local hostport="$1"; shift
    local host="${hostport%%:*}"
    local port="${hostport##*:}"
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$port" "root@$host" "$@"
}

echo "=== Verifying connectivity ==="
for node in "$HEAD_SSH" "${WORKERS[@]}"; do
    echo -n "  $node: "
    ssh_cmd "$node" "hostname && nvidia-smi --query-gpu=name --format=csv,noheader | head -1" || { echo "FAILED"; exit 1; }
done

echo ""
echo "=== Installing Ray on head node ==="
ssh_cmd "$HEAD_SSH" 'bash -s' << 'HEADSETUP'
pip install -q ray[default] 2>/dev/null || pip3 install -q ray[default] 2>/dev/null
# Get the node's internal IP for Ray
INTERNAL_IP=$(hostname -I | awk '{print $1}')
echo "Head internal IP: $INTERNAL_IP"

# Start Ray head
ray stop --force 2>/dev/null || true
ray start --head --port=6379 --dashboard-host=0.0.0.0 --node-ip-address="$INTERNAL_IP"
echo "HEAD_IP=$INTERNAL_IP"
HEADSETUP

# Extract head IP
HEAD_IP=$(ssh_cmd "$HEAD_SSH" "hostname -I | awk '{print \$1}'")
echo "Head node IP: $HEAD_IP"

echo ""
echo "=== Setting up workers ==="
for i in "${!WORKERS[@]}"; do
    echo "--- Worker $((i+1)): ${WORKERS[$i]} ---"
    ssh_cmd "${WORKERS[$i]}" "bash -s" << WORKERSETUP
pip install -q ray[default] 2>/dev/null || pip3 install -q ray[default] 2>/dev/null
ray stop --force 2>/dev/null || true
INTERNAL_IP=\$(hostname -I | awk '{print \$1}')
ray start --address=$HEAD_IP:6379 --node-ip-address="\$INTERNAL_IP"
WORKERSETUP
done

echo ""
echo "=== Verifying Ray cluster ==="
ssh_cmd "$HEAD_SSH" "ray status"

echo ""
echo "=== Pre-downloading model on all nodes ==="
for node in "$HEAD_SSH" "${WORKERS[@]}"; do
    echo "  Starting model download on $node..."
    ssh_cmd "$node" 'bash -s' << 'DLMODEL'
nohup python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-30B-A3B')
print('Model download complete')
" > /root/model_download.log 2>&1 &
echo "Download started in background (PID $!)"
DLMODEL
done

echo ""
echo "=== Cluster ready ==="
echo "Head: $HEAD_SSH (IP: $HEAD_IP)"
for i in "${!WORKERS[@]}"; do
    echo "Worker $((i+1)): ${WORKERS[$i]}"
done
echo ""
echo "Run the experiment with:"
echo "  ssh -p PORT root@HOST 'cd /root/deterministic_serving_stack && ./scripts/ci/d6_multinode_determinism.sh'"
