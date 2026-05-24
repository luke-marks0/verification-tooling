#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.core.common.jsonschema_compat import DefaultValidator, DEFAULT_VALIDATOR_NAME

SCHEMA_RE = re.compile(r"^[a-z_]+\.v\d+\.schema\.json$")


def main() -> int:
    schema_files = sorted(pathlib.Path("modules/core/schemas").glob("*.schema.json"))
    if not schema_files:
        print("No schema files found", file=sys.stderr)
        return 1

    for schema_file in schema_files:
        if not SCHEMA_RE.match(schema_file.name):
            print(f"Invalid schema filename: {schema_file.name}", file=sys.stderr)
            return 1

        try:
            data = json.loads(schema_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON in schema {schema_file}: {exc}", file=sys.stderr)
            return 1

        try:
            DefaultValidator.check_schema(data)
        except Exception as exc:  # pragma: no cover - third-party exception tree
            print(f"Schema does not pass Draft 2020-12 meta-validation: {schema_file}: {exc}", file=sys.stderr)
            return 1

        for key in ("$schema", "title", "type"):
            if key not in data:
                print(f"Schema missing '{key}': {schema_file}", file=sys.stderr)
                return 1

    print(f"Validated {len(schema_files)} schema file(s) against {DEFAULT_VALIDATOR_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
