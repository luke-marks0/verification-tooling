#!/usr/bin/env bash
# Poll Lambda Cloud until a batch-invariance-capable GPU (CC >= 9.0) is available, then launch it.
# Usage: scripts/deploy/lambda/grab_instance.sh
set -euo pipefail

WANTED_TYPES=("gpu_1x_gh200" "gpu_1x_h100_pcie" "gpu_1x_h100_sxm5")
SSH_KEY_NAME="macbook 2025"
INSTANCE_NAME="det-serving-bi"
POLL_INTERVAL=30

echo "Polling Lambda Cloud for available instances: ${WANTED_TYPES[*]}"
echo "Poll interval: ${POLL_INTERVAL}s"
echo ""

while true; do
    AVAIL=$(curl -s -u "$LAMBDALABS_API_KEY:" https://cloud.lambdalabs.com/api/v1/instance-types)

    for itype in "${WANTED_TYPES[@]}"; do
        REGIONS=$(echo "$AVAIL" | python3 -c "
import json, sys
data = json.load(sys.stdin)['data']
info = data.get('${itype}', {})
avail = info.get('regions_with_capacity_available', [])
for r in avail:
    print(r['name'])
" 2>/dev/null || true)

        if [ -n "$REGIONS" ]; then
            REGION=$(echo "$REGIONS" | head -1)
            echo "$(date): FOUND ${itype} in ${REGION} — launching..."

            RESULT=$(curl -s -u "$LAMBDALABS_API_KEY:" -X POST \
                https://cloud.lambdalabs.com/api/v1/instance-operations/launch \
                -H 'Content-Type: application/json' \
                -d "{
                    \"region_name\": \"${REGION}\",
                    \"instance_type_name\": \"${itype}\",
                    \"ssh_key_names\": [\"${SSH_KEY_NAME}\"],
                    \"name\": \"${INSTANCE_NAME}\"
                }")

            ERROR=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('error',{}).get('code',''))" 2>/dev/null || true)

            if [ -z "$ERROR" ]; then
                INSTANCE_ID=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data']['instance_ids'][0])")
                echo "Launched! Instance ID: ${INSTANCE_ID}"
                echo ""
                echo "Waiting for IP..."

                for _ in $(seq 1 60); do
                    sleep 5
                    INFO=$(curl -s -u "$LAMBDALABS_API_KEY:" "https://cloud.lambdalabs.com/api/v1/instances/${INSTANCE_ID}")
                    IP=$(echo "$INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'].get('ip',''))" 2>/dev/null || true)
                    STATUS=$(echo "$INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'].get('status',''))" 2>/dev/null || true)

                    if [ -n "$IP" ] && [ "$STATUS" = "active" ]; then
                        echo ""
                        echo "=== Instance Ready ==="
                        echo "Type:   ${itype}"
                        echo "Region: ${REGION}"
                        echo "IP:     ${IP}"
                        echo "ID:     ${INSTANCE_ID}"
                        echo ""
                        echo "SSH:    ssh ubuntu@${IP}"
                        echo ""
                        # Write connection info for other scripts
                        echo "${IP}" > /tmp/lambda_instance_ip
                        echo "${INSTANCE_ID}" > /tmp/lambda_instance_id
                        exit 0
                    fi
                    printf "."
                done
                echo "Timed out waiting for instance to become active"
                exit 1
            else
                echo "$(date): Launch failed (${ERROR}), retrying..."
            fi
        fi
    done

    printf "$(date '+%H:%M:%S') no availability\r"
    sleep "$POLL_INTERVAL"
done
