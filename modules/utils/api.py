"""Shared determinism helpers — stable public API. See ``README.md``.

Re-exports the canonical-JSON / digest / schema-validation primitives the whole
artifact spine relies on (``modules.core.common``). Cloud provisioning and the
replay-server routine live in ``scripts/deploy/*`` (shell) and are documented in the
README rather than wrapped here.
"""
from __future__ import annotations

from modules.core.common.contracts import ValidationError, validate_with_schema
from modules.core.common.deterministic import (
    canonical_json_bytes,
    canonical_json_text,
    compute_bundle_digest,
    compute_lockfile_digest,
    sha256_file,
    sha256_hex,
    sha256_prefixed,
    utc_now_iso,
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
