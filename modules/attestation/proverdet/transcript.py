"""Verifier-side transcript log.

Schema: see schemas/verifier_transcript_entry.v1.schema.json. Backed by
JsonlLog with schema validation enabled — every entry is checked before
write so the verdict engine only ever reads well-typed data.
"""

from __future__ import annotations

from modules.attestation.proverdet._jsonl_log import JsonlLog


class TranscriptLog(JsonlLog):
    SCHEMA_NAME = "verifier_transcript_entry.v1.schema.json"
