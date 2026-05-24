from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from modules.attestation.proverdet.erasure import (
    ErasureRoundLog,
    HmacErasureBackend,
    run_erasure,
    verify_round_log,
)
from modules.attestation.proverdet.wire import ErasureSpec


def _spec(rounds: int = 4, *, seed: str = "deadbeefdeadbeef") -> ErasureSpec:
    return ErasureSpec(challenge_seed=seed, deadline_ms=1000, rounds=rounds)


class TestHonestErasure(unittest.TestCase):
    def test_honest_run_passes_all_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "erasure.jsonl"
            ev = run_erasure(
                _spec(rounds=8),
                log_path=log_path,
                backend=HmacErasureBackend(),
            )
            self.assertEqual(ev.rounds, 8)
            self.assertEqual(ev.passed, 8)
            self.assertEqual(ev.log_path, str(log_path))

    def test_honest_run_writes_jsonl_with_one_entry_per_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "erasure.jsonl"
            run_erasure(
                _spec(rounds=5),
                log_path=log_path,
                backend=HmacErasureBackend(),
            )
            lines = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 5)
            for i, line in enumerate(lines):
                entry = json.loads(line)
                self.assertEqual(entry["round"], i)
                self.assertTrue(entry["passed"])
                self.assertIn("expected_hex", entry)
                self.assertIn("response_hex", entry)
                self.assertEqual(entry["expected_hex"], entry["response_hex"])

    def test_log_replays_to_same_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "erasure.jsonl"
            run_erasure(
                _spec(rounds=3),
                log_path=log_path,
                backend=HmacErasureBackend(),
            )
            entries = [
                ErasureRoundLog.model_validate_json(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            ok = verify_round_log(_spec(rounds=3), entries)
            self.assertTrue(ok)


class TestDishonestErasure(unittest.TestCase):
    def test_corrupted_response_fails_at_least_one_round(self) -> None:
        spec = _spec(rounds=4)
        # The "lying" backend returns a fixed all-zeros response for every
        # round — which only matches the honest expected output by accident.
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "erasure.jsonl"
            ev = run_erasure(spec, log_path=log_path, backend=_LiarBackend())
            self.assertEqual(ev.rounds, spec.rounds)
            self.assertLess(ev.passed, ev.rounds)

    def test_verifier_rejects_tampered_log(self) -> None:
        spec = _spec(rounds=3)
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "erasure.jsonl"
            run_erasure(spec, log_path=log_path, backend=HmacErasureBackend())
            # Tamper: flip the response_hex on the first entry.
            lines = log_path.read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0])
            first["response_hex"] = "00" * (len(first["response_hex"]) // 2)
            entries = [ErasureRoundLog.model_validate(first)] + [
                ErasureRoundLog.model_validate_json(line) for line in lines[1:]
            ]
            self.assertFalse(verify_round_log(spec, entries))


class _LiarBackend:
    """Always returns 32 zero bytes regardless of seed/round."""

    def respond(self, seed: bytes, round_index: int) -> bytes:
        return b"\x00" * 32


if __name__ == "__main__":
    unittest.main()
