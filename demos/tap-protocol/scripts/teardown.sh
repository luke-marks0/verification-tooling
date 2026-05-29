#!/usr/bin/env bash
# Destroy the vast instance recorded in .last_instance.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

INSTANCE_FILE="$DEMO_DIR/.last_instance"
if [ ! -f "$INSTANCE_FILE" ]; then
    echo "no .last_instance found at $INSTANCE_FILE" >&2
    exit 1
fi

INSTANCE_ID=$(cat "$INSTANCE_FILE")
echo "destroying vast instance $INSTANCE_ID..."
vastai destroy instance "$INSTANCE_ID"
rm -f "$INSTANCE_FILE"
