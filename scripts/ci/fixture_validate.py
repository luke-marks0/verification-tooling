#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.core.common.jsonschema_compat import DefaultValidator

SCHEMA_DIR = Path("modules/core/schemas")
POS_DIR = Path("tests/fixtures/positive")
NEG_DIR = Path("tests/fixtures/negative")
POS_NAME_RE = re.compile(r"^(?P<schema>[a-z_]+\.v\d+)\.example\.json$")
NEG_NAME_RE = re.compile(r"^(?P<schema>[a-z_]+\.v\d+)__.+\.invalid\.json$")


def _validator(schema_file: Path) -> DefaultValidator:
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    DefaultValidator.check_schema(schema)
    return DefaultValidator(schema)


def _validate_positive() -> tuple[int, list[str]]:
    errors: list[str] = []
    files = sorted(POS_DIR.glob("*.json"))
    if not files:
        return (0, ["No positive fixture files found"])

    for fixture in files:
        match = POS_NAME_RE.match(fixture.name)
        if not match:
            errors.append(f"Unexpected positive fixture name: {fixture}")
            continue

        schema_name = f"{match.group('schema')}.schema.json"
        schema_file = SCHEMA_DIR / schema_name
        if not schema_file.exists():
            errors.append(f"Missing schema for positive fixture {fixture}: {schema_file}")
            continue

        data = json.loads(fixture.read_text(encoding="utf-8"))
        validator = _validator(schema_file)
        v_errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
        if v_errors:
            errors.append(f"Positive fixture failed validation: {fixture}: {v_errors[0].message}")

    return (len(files), errors)


def _validate_negative() -> tuple[int, list[str]]:
    errors: list[str] = []
    files = sorted(NEG_DIR.glob("*.json"))
    if not files:
        return (0, ["No negative fixture files found"])

    for fixture in files:
        match = NEG_NAME_RE.match(fixture.name)
        if not match:
            errors.append(f"Unexpected negative fixture name: {fixture}")
            continue

        schema_name = f"{match.group('schema')}.schema.json"
        schema_file = SCHEMA_DIR / schema_name
        if not schema_file.exists():
            errors.append(f"Missing schema for negative fixture {fixture}: {schema_file}")
            continue

        data = json.loads(fixture.read_text(encoding="utf-8"))
        validator = _validator(schema_file)
        v_errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
        if not v_errors:
            errors.append(f"Negative fixture unexpectedly passed validation: {fixture}")

    return (len(files), errors)


def main() -> int:
    pos_count, pos_errors = _validate_positive()
    neg_count, neg_errors = _validate_negative()

    all_errors = [*pos_errors, *neg_errors]
    if all_errors:
        for err in all_errors:
            print(err, file=sys.stderr)
        return 1

    print(f"Validated {pos_count} positive fixture(s) and {neg_count} negative fixture(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
