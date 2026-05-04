"""Verdict engine for the verifier (stub).

Phase 8 will replace this stub with three real signals (replay
correctness, compute budget, bandwidth) and a combiner. Until then,
emit_verdict returns the empty `unknown` verdict for any transcript so
downstream code (verdict CLI, demo runner) can be wired today and
sharpened as the signals come in.
"""

from __future__ import annotations

from pathlib import Path


def emit_verdict(transcript_path: Path) -> dict[str, object]:
    """Read the transcript, emit a verdict.

    Phase 3.4 stub: always returns the empty `unknown` verdict regardless
    of contents. The signature stays stable through Phase 8 — Task 8.3
    extends the call site to also pass `traffic_digest_path`.
    """
    # Touch the transcript so we surface a clear error if the path is
    # wrong (rather than silently returning unknown for a missing file).
    if not Path(transcript_path).exists():
        raise FileNotFoundError(f"transcript not found: {transcript_path}")

    return {"verdict": "unknown", "reasons": []}
