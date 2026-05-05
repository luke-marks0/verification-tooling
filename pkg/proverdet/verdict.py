"""Verdict engine for the verifier.

Three signals plus a combiner:

  * `replay_correctness` — every recorded /replay/verdict/{id} entry
    must be status 200. Reads transcript entries.
  * `compute_budget` — sum(observed_flops) must not exceed
    (1 + tolerance) * sum(claimed_flops). Reads `summaries`. Phase 8.3
    accepts an optional `workload_summary` adding workload-internal
    counters (the mixed_lora gradient-step fingerprint).
  * `bandwidth_signal` — traffic_size must not exceed claimed_artifact_bytes
    (zero tolerance: the verifier's tap gives ground-truth bytes; any extra
    byte is unattributed traffic). Catches lora_loading.

Combiner: any failed signal fires `"training_or_exfil"` with reasons
concatenated; otherwise `"inference"`. We intentionally don't preserve
"unknown" — Phase 8.3's binary verdict is the demo's headline output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SignalResult:
    passed: bool
    reasons: list[str]


# ---- replay_correctness (Task 8.1) -----------------------------------


def replay_correctness(transcript_entries: list[dict[str, object]]) -> SignalResult:
    """`passed` iff every recorded /replay/verdict/{id} entry is status 200."""
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
    if not saw_any or not failures:
        return SignalResult(passed=True, reasons=[])
    return SignalResult(passed=False, reasons=failures)


# ---- compute_budget (Task 8.2 + 8.3 workload_summary extension) ------


def compute_budget(
    summaries: list[dict[str, object]],
    *,
    workload_summary: dict[str, object] | None = None,
    tolerance: float = 0.10,
) -> SignalResult:
    """`passed` iff observed_flops <= (1 + tolerance) * claimed_flops.

    `summaries` are the scheduler-emitted /graph and /replay_evidence
    summaries. `workload_summary` (Task 8.3) is an out-of-band record
    from the prover's /workload/stop response carrying the workload's
    own claimed_flops_total + observed_flops_total — necessary to detect
    mixed_lora's gradient-step cheating, which doesn't show up in the
    scheduler's /replay challenges.
    """
    claimed_total = 0
    observed_total = 0
    for s in summaries:
        kind = s.get("kind")
        if kind == "graph":
            v = s.get("claimed_flops_total", 0)
            if isinstance(v, int):
                claimed_total += v
        elif kind == "replay_evidence":
            v = s.get("observed_flops", 0)
            if isinstance(v, int):
                observed_total += v
    if workload_summary is not None:
        c = workload_summary.get("claimed_flops_total", 0)
        o = workload_summary.get("observed_flops_total", 0)
        if isinstance(c, int):
            claimed_total += c
        if isinstance(o, int):
            observed_total += o

    if claimed_total == 0 and observed_total == 0:
        return SignalResult(passed=True, reasons=[])

    threshold = (1.0 + tolerance) * claimed_total
    if observed_total > threshold:
        return SignalResult(
            passed=False,
            reasons=[
                f"compute budget exceeded: observed_flops={observed_total} "
                f"vs claimed_flops={claimed_total} (tolerance={tolerance:.2f})"
            ],
        )
    return SignalResult(passed=True, reasons=[])


# ---- bandwidth_signal (Task 8.3) -------------------------------------


def bandwidth_signal(
    traffic_size: int,
    claimed_artifact_bytes: int,
    *,
    tolerance: float = 0.0,
) -> SignalResult:
    """`passed` iff traffic_size <= (1 + tolerance) * claimed_artifact_bytes.

    Default tolerance is **zero**: the verifier observes ground-truth bytes
    on the wire from a trusted tap, and the prover's claim is an integer
    byte count — there is no float drift, no encoding overhead, and no
    timing noise to absorb. Any extra byte is unattributed traffic and
    should fire the signal. The `tolerance` knob is kept for callers who
    want a soft margin during early bring-up; production should leave it
    at 0.

    When `claimed_artifact_bytes == 0` we can't draw a conclusion (no
    baseline) — return passed and rely on the other signals. Otherwise the
    predicate fires when traffic exceeds the envelope; that's the
    lora_loading fingerprint.
    """
    if claimed_artifact_bytes <= 0:
        return SignalResult(passed=True, reasons=[])
    threshold = int((1.0 + tolerance) * claimed_artifact_bytes)
    if traffic_size > threshold:
        return SignalResult(
            passed=False,
            reasons=[
                f"bandwidth exceeded: traffic_size={traffic_size} bytes vs "
                f"claimed_artifact_bytes={claimed_artifact_bytes} "
                f"(tolerance={tolerance:.2f})"
            ],
        )
    return SignalResult(passed=True, reasons=[])


# ---- combiner + emit_verdict -----------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_traffic_size(traffic_digest_path: Path | None) -> int:
    """Derive the verifier's recorded traffic size in bytes.

    `traffic_digest_path` itself only carries the digest. The matching
    payload lives at `traffic.bin` in the same directory; we measure that.
    """
    if traffic_digest_path is None:
        return 0
    bin_path = traffic_digest_path.parent / "traffic.bin"
    if not bin_path.exists():
        return 0
    return bin_path.stat().st_size


def emit_verdict(
    transcript_path: Path,
    traffic_digest_path: Path | None = None,
    *,
    workload_summary_path: Path | None = None,
    tolerance: float = 0.10,
) -> dict[str, object]:
    """Read transcript + sidecars, run all signals, emit a binary verdict."""
    if not Path(transcript_path).exists():
        raise FileNotFoundError(f"transcript not found: {transcript_path}")

    transcript_entries = _read_jsonl(Path(transcript_path))
    summaries = _read_jsonl(Path(transcript_path).parent / "summaries.jsonl")
    workload_summary: dict[str, object] | None = None
    if workload_summary_path is not None and Path(workload_summary_path).exists():
        workload_summary = json.loads(Path(workload_summary_path).read_text(encoding="utf-8"))

    # Bandwidth-side claim: the benign inference workload sets
    # claimed_flops == bytes_on_wire by construction, so we reuse the same
    # total as the bandwidth baseline. lora_loading's downloaded bytes are
    # NOT recorded as tasks, so they appear as a positive gap.
    claimed_artifact_bytes = 0
    for s in summaries:
        if s.get("kind") == "graph":
            v = s.get("claimed_flops_total", 0)
            if isinstance(v, int):
                claimed_artifact_bytes += v
    if workload_summary is not None:
        c = workload_summary.get("claimed_flops_total", 0)
        if isinstance(c, int):
            claimed_artifact_bytes += c

    traffic_size = _read_traffic_size(
        Path(traffic_digest_path) if traffic_digest_path is not None else None
    )

    correctness = replay_correctness(transcript_entries)
    budget = compute_budget(summaries, workload_summary=workload_summary, tolerance=tolerance)
    # Bandwidth runs at zero tolerance regardless of `tolerance` — bytes are
    # bytes; the verifier's tap is trusted; there is no noise envelope to
    # widen. The `tolerance` arg here only affects compute_budget.
    bandwidth = bandwidth_signal(traffic_size, claimed_artifact_bytes)

    reasons: list[str] = []
    if not correctness.passed:
        reasons.extend(correctness.reasons)
    if not budget.passed:
        reasons.extend(budget.reasons)
    if not bandwidth.passed:
        reasons.extend(bandwidth.reasons)

    if reasons:
        return {"verdict": "training_or_exfil", "reasons": reasons}
    return {"verdict": "inference", "reasons": []}
