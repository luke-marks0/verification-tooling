"""PoSE-style erasure protocol — HMAC-over-(seed, round_index).

The verifier issues a 256-bit `challenge_seed` and a round count. For each
round, both parties compute `expected = HMAC-SHA256(seed, round_bytes)`.
An honest prover returns `expected`; a dishonest one returns whatever
bytes it has lying around. Rounds where `response == expected` count as
passed.

We don't bother with the full PoSE memory-fill phase here — that requires
a real GPU node and is exercised by the `memory_wipe` experiment on the
`experiments` branch. For the
prover-verifier demo, the honest-path protocol is enough to give the
verdict engine a trustworthy "we ran K rounds and N passed" signal; the
dishonest path is exercised by an adversarial workload in Phase 7.

The on-disk log is JSONL: one `ErasureRoundLog` per round. Phase 6.4's
verifier replays the log against the request's `challenge_seed` to detect
post-hoc tampering.
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from modules.attestation.proverdet.wire import ErasureEvidence, ErasureSpec, HexString


class ErasureBackend(Protocol):
    def respond(self, seed: bytes, round_index: int) -> bytes: ...


class HmacErasureBackend:
    """Honest backend: returns HMAC-SHA256(seed, round_bytes)."""

    def respond(self, seed: bytes, round_index: int) -> bytes:
        return _expected(seed, round_index)


class ErasureRoundLog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    round: int = Field(ge=0)
    expected_hex: HexString
    response_hex: HexString
    passed: bool


def _expected(seed: bytes, round_index: int) -> bytes:
    return hmac.new(seed, round_index.to_bytes(8, "big"), hashlib.sha256).digest()


def run_erasure(
    spec: ErasureSpec,
    *,
    log_path: Path,
    backend: ErasureBackend,
) -> ErasureEvidence:
    """Run `spec.rounds` HMAC challenges, write the log, return evidence."""
    seed = bytes.fromhex(spec.challenge_seed)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    passed = 0
    with log_path.open("w", encoding="utf-8") as f:
        for r in range(spec.rounds):
            expected = _expected(seed, r)
            response = backend.respond(seed, r)
            ok = response == expected
            entry = ErasureRoundLog(
                round=r,
                expected_hex=expected.hex(),
                response_hex=response.hex(),
                passed=ok,
            )
            f.write(entry.model_dump_json() + "\n")
            if ok:
                passed += 1

    return ErasureEvidence(
        rounds=spec.rounds,
        passed=passed,
        log_path=str(log_path),
    )


def verify_round_log(spec: ErasureSpec, entries: list[ErasureRoundLog]) -> bool:
    """Recompute expected per round; reject any drift from the log."""
    if len(entries) != spec.rounds:
        return False
    seed = bytes.fromhex(spec.challenge_seed)
    for r, entry in enumerate(entries):
        if entry.round != r:
            return False
        if entry.expected_hex != _expected(seed, r).hex():
            return False
        actual_pass = entry.expected_hex == entry.response_hex
        if entry.passed != actual_pass:
            return False
    return True
