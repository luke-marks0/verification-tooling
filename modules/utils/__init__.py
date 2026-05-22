"""Utilities capability — canonical JSON, digests, schema validation."""
from modules.utils.api import (
    ValidationError,
    canonical_json_bytes,
    canonical_json_text,
    compute_bundle_digest,
    compute_lockfile_digest,
    sha256_file,
    sha256_hex,
    sha256_prefixed,
    utc_now_iso,
    validate_with_schema,
)

__all__ = [
    "canonical_json_bytes",
    "canonical_json_text",
    "sha256_hex",
    "sha256_prefixed",
    "sha256_file",
    "compute_lockfile_digest",
    "compute_bundle_digest",
    "validate_with_schema",
    "ValidationError",
    "utc_now_iso",
]
