"""Prover-side capture log.

Lifts the *pattern* from modules/inference/server/main.py's CaptureLog (append-only JSONL,
monotonic seq, threadsafe). The deliberately-different class name avoids
confusion with the existing CaptureLog. Backed by JsonlLog without schema
validation — the prover capture log is a debugging aid, not consumed by
the verdict engine, so we don't need to pay the validation cost on each
append.
"""

from __future__ import annotations

from modules.attestation.proverdet._jsonl_log import JsonlLog


class ProverCaptureLog(JsonlLog):
    SCHEMA_NAME = None
