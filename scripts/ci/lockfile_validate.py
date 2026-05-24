#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.core.common.contracts import ValidationError, validate_with_schema
from modules.core.common.deterministic import compute_lockfile_digest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate lockfile schema and canonical digest")
    parser.add_argument("--lockfile", required=True, help="Lockfile JSON path")
    args = parser.parse_args()

    lockfile_path = Path(args.lockfile)
    data = json.loads(lockfile_path.read_text(encoding="utf-8"))

    try:
        validate_with_schema("lockfile.v1.schema.json", data)
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    expected = compute_lockfile_digest(data)
    actual = data["canonicalization"]["lockfile_digest"]
    if actual != expected:
        print(
            f"Lockfile digest mismatch: expected={expected} actual={actual} file={lockfile_path}",
            file=sys.stderr,
        )
        return 1

    print(f"Lockfile valid and canonical digest verified: {lockfile_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
