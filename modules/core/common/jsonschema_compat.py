from __future__ import annotations

try:
    from jsonschema import Draft202012Validator as DefaultValidator  # noqa: F401  (availability probe)
    DEFAULT_VALIDATOR_NAME = "Draft202012Validator"
except ImportError:  # pragma: no cover - depends on runner image
    DEFAULT_VALIDATOR_NAME = "Draft7Validator"

