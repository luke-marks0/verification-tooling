#!/usr/bin/env bash
# Resolve the /nix/store/<hash>-entrypoint/bin/entrypoint path for the
# vast-test image in ghcr. The hash changes every rebuild so the launcher
# must fetch it dynamically.
# Verbatim shape from /home/jon/.claude/CLAUDE.md vast section.
set -euo pipefail

IMAGE_REPO="${IMAGE_REPO:-derpyplops/deterministic-serving}"
IMAGE_TAG="${IMAGE_TAG:-vast-test}"

TOK=$(gh auth token)
B64TOK=$(echo -n "$TOK" | base64)

DIGEST=$(curl -sL -H "Authorization: Bearer $B64TOK" \
    -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
    "https://ghcr.io/v2/${IMAGE_REPO}/manifests/${IMAGE_TAG}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['config']['digest'])")

ENTRY=$(curl -sL -H "Authorization: Bearer $B64TOK" \
    "https://ghcr.io/v2/${IMAGE_REPO}/blobs/${DIGEST}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['config']['Cmd'][0])")

echo "$ENTRY"
