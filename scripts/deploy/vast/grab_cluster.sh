#!/usr/bin/env bash
# Provision 4× H100 SXM instances on vast.ai for multi-node determinism testing.
# Usage: ./grab_cluster.sh [--image IMAGE] [--disk DISK_GB]
#
# Outputs node IPs/ports to cluster.env for use by setup_cluster.sh.
set -euo pipefail

IMAGE="${1:---image ghcr.io/derpyplops/deterministic-serving:multinode}"
DISK="${2:-100}"
NUM_WORKERS=3

echo "=== Searching for H100 SXM offers ==="
vastai search offers \
  'gpu_name=H100_SXM num_gpus=1 cuda_vers>=12.0 reliability>0.90 inet_down>300 disk_space>80' \
  -o 'dph' --raw | python3 -c "
import sys, json
offers = json.load(sys.stdin)
for o in offers[:10]:
    print(f'  ID:{o[\"id\"]:>8}  \${o.get(\"dph_total\",0):.2f}/hr  {o.get(\"gpu_name\",\"?\")}  {o.get(\"geolocation\",\"?\")}  reliability:{o.get(\"reliability\",0):.2f}')
"

echo ""
echo "=== Creating head node (Node 0) ==="
echo "Using bare ubuntu for OCI build node. Workers will use the GHCR image."

read -rp "Enter offer ID for head node: " HEAD_OFFER
HEAD_ID=$(vastai create instance "$HEAD_OFFER" --image ubuntu:22.04 --disk "$DISK" --raw | python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
echo "Head node contract: $HEAD_ID"

echo ""
echo "=== Creating $NUM_WORKERS worker nodes ==="
WORKER_IDS=()
for i in $(seq 1 $NUM_WORKERS); do
    read -rp "Enter offer ID for worker $i: " WORKER_OFFER
    WID=$(vastai create instance "$WORKER_OFFER" --image "$IMAGE" --disk 50 --raw | python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
    WORKER_IDS+=("$WID")
    echo "  Worker $i contract: $WID"
done

# Write cluster config
cat > cluster.env << EOF
HEAD_CONTRACT=$HEAD_ID
WORKER_CONTRACTS=${WORKER_IDS[*]}
EOF

echo ""
echo "=== Cluster provisioned ==="
echo "Written to cluster.env"
echo "Wait for instances to boot, then run: ./setup_cluster.sh"
echo ""
echo "Check status with: vastai show instances"
