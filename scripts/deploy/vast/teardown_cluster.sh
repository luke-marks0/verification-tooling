#!/usr/bin/env bash
# Destroy all vast.ai instances from this cluster.
# Usage: ./teardown_cluster.sh [contract_id ...]
#
# If no args, destroys ALL running instances (use with caution).
set -euo pipefail

if [ $# -gt 0 ]; then
    for id in "$@"; do
        echo "Destroying instance $id..."
        vastai destroy instance "$id"
    done
else
    echo "=== Destroying ALL vast.ai instances ==="
    vastai show instances --raw | python3 -c "
import sys, json
instances = json.load(sys.stdin)
for inst in instances:
    print(f'  Destroying {inst[\"id\"]} ({inst.get(\"actual_status\", \"unknown\")})...')
"
    read -rp "Confirm destroy all? [y/N] " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        vastai show instances --raw | python3 -c "
import sys, json
for inst in json.load(sys.stdin):
    print(inst['id'])
" | while read -r id; do
            vastai destroy instance "$id"
        done
        echo "All instances destroyed."
    else
        echo "Aborted."
    fi
fi
