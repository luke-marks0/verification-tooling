"""Verdict engine for the verifier.

Three signals (replay correctness, compute budget, bandwidth) feed a
combiner that emits the final verdict. Phase 8.1 lands `replay_correctness`
+ the `SignalResult` shape; 8.2/8.3 add the rest and the combiner.

`emit_verdict` is kept stable so the verdict CLI doesn't churn — the
combiner replaces this body in 8.3, but the function signature stays
(transcript_path, [traffic_digest_path]) -> dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SignalResult:
    passed: bool
    reasons: list[str]


def replay_correctness(transcript_entries: list[dict[str, object]]) -> SignalResult:
    """`passed` iff every recorded /replay/verdict entry is status 200.

    Failed entries surface a reason naming the replay_id (extracted from
    the endpoint suffix: `/replay/verdict/<replay_id>` is the convention
    the scheduler writes).
    """
    failures: list[str] = []
    saw_any = False
    for e in transcript_entries:
        endpoint = e.get("endpoint")
        if not isinstance(endpoint, str):
            continue
        if e.get("direction") != "received":
            continue
        if not endpoint.startswith("/replay/verdict/"):
            continue
        saw_any = True
        if e.get("status_code") == 200:
            continue
        replay_id = endpoint[len("/replay/verdict/") :]
        failures.append(f"replay {replay_id} failed: status_code={e.get('status_code')}")
    if not saw_any:
        # No verdicts to check ⇒ no evidence of failure. Phase 8.3's
        # combiner can flip the final verdict to "unknown" when the
        # transcript is too thin to draw a conclusion.
        return SignalResult(passed=True, reasons=[])
    if failures:
        return SignalResult(passed=False, reasons=failures)
    return SignalResult(passed=True, reasons=[])


def _read_transcript(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"transcript not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def emit_verdict(transcript_path: Path) -> dict[str, object]:
    """Read the transcript, emit a verdict.

    Phase 8.1 wires `replay_correctness` only. Phase 8.2 adds compute
    budget; Phase 8.3 adds bandwidth and the final combiner. Until 8.3
    the verdict is "unknown" when no signal trips and "training_or_exfil"
    when any does.
    """
    entries = _read_transcript(transcript_path)
    correctness = replay_correctness(entries)

    if correctness.passed:
        return {"verdict": "unknown", "reasons": []}
    return {"verdict": "training_or_exfil", "reasons": list(correctness.reasons)}
