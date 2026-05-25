#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> int:
    blockers_path = Path("conformance/RELEASE_BLOCKERS.json")
    blockers = json.loads(blockers_path.read_text(encoding="utf-8"))
    required = blockers.get("required_conformance_ids", [])
    if not required:
        print("No required_conformance_ids configured", file=sys.stderr)
        return 1

    result_dir = Path(".ci-results/conformance")
    missing: list[str] = []
    for cid in required:
        marker = result_dir / f"{cid}.pass"
        if not marker.exists() or marker.read_text(encoding="utf-8").strip() != "PASS":
            missing.append(cid)

    if missing:
        print("Release blockers not satisfied:", file=sys.stderr)
        for cid in missing:
            print(f"  - {cid}", file=sys.stderr)
        return 1

    print(f"All {len(required)} release blocker conformance IDs passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
