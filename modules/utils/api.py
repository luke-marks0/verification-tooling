"""Shared determinism helpers — stable public API. See ``README.md``.

Re-exports the canonical-JSON / digest / schema-validation primitives the whole
artifact spine relies on (``pkg.common``). Cloud provisioning and the
replay-server routine live in ``deploy/*`` (shell) and are documented in the
README rather than wrapped here.
"""
from __future__ import annotations

from pkg.common.contracts import ValidationError, validate_with_schema
from pkg.common.deterministic import (
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
