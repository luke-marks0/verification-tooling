from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modules.core.common.jsonschema_compat import DefaultValidator

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


class ValidationError(Exception):
    pass


def _load_schema(schema_name: str) -> dict[str, Any]:
    path = SCHEMA_DIR / schema_name
    if not path.exists():
        raise ValidationError(f"Missing schema: {path}")
    schema = json.loads(path.read_text(encoding="utf-8"))
    DefaultValidator.check_schema(schema)
    return schema


def validate_with_schema(schema_name: str, data: Any) -> None:
    schema = _load_schema(schema_name)
    validator = DefaultValidator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        loc = "$"
        for part in first.path:
            if isinstance(part, int):
                loc += f"[{part}]"
            else:
                loc += f".{part}"
        raise ValidationError(f"{schema_name}: {loc}: {first.message}")
