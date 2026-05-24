#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.core.common.contracts import ValidationError, validate_with_schema
from modules.core.common.deterministic import canonical_json_text, sha256_prefixed


def deterministic_dispatch(manifest: dict[str, Any], replicas: list[str]) -> list[dict[str, Any]]:
    validate_with_schema("manifest.v1.schema.json", manifest)
    if not replicas:
        raise ValidationError("At least one replica is required")

    rack_count = manifest.get("hardware_profile", {}).get("topology", {}).get("rack_count", 1)
    algorithm = manifest.get("deterministic_dispatcher", {}).get("algorithm", "round_robin_hash")

    assignments: list[dict[str, Any]] = []
    for seq, req in enumerate(manifest["requests"]):
        if algorithm == "sequence_map":
            idx = seq % len(replicas)
        else:
            digest = sha256_prefixed(canonical_json_text({"id": req["id"], "seq": seq}).encode("utf-8"))
            idx = int(digest.split(":", 1)[1][:8], 16) % len(replicas)

        replica_id = replicas[idx]
        rack_id = idx % rack_count
        assignments.append(
            {
                "sequence": seq,
                "request_id": req["id"],
                "replica_id": replica_id,
                "rack_id": rack_id,
            }
        )
    return assignments


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic request dispatcher")
    parser.add_argument("--manifest", required=True, help="Manifest path")
    parser.add_argument("--replicas", required=True, help="Comma-separated replica IDs")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    replicas = [item.strip() for item in args.replicas.split(",") if item.strip()]

    assignments = deterministic_dispatch(manifest, replicas)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(canonical_json_text(assignments), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(str(exc))
        raise SystemExit(1)
