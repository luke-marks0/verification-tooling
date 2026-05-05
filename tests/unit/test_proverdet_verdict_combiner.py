"""Combiner tests for `emit_verdict` — the Phase 8.3 binary verdict.

End-to-end-ish: write a real transcript + summaries + (optional) workload
summary + (optional) traffic.bin to disk, call `emit_verdict`, assert the
verdict and reason structure. Each scenario isolates one signal so a
regression in any single signal surfaces as a focused failure.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


def _write_jsonl(path: Path, items: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(it, sort_keys=True) + "\n" for it in items),
        encoding="utf-8",
    )


def _verdict_entry(seq: int, replay_id: str, *, status: int) -> dict[str, object]:
    return {
        "seq": seq,
        "direction": "received",
        "endpoint": f"/replay/verdict/{replay_id}",
        "timestamp": "2026-05-04T00:00:00Z",
        "payload_digest": "sha256:" + "0" * 64,
        "status_code": status,
    }


class TestVerdictCombiner(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.transcript = self.dir / "transcript.jsonl"
        self.summaries = self.dir / "summaries.jsonl"
        self.transcript.write_text("", encoding="utf-8")
        self.summaries.write_text("", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _emit(
        self,
        *,
        traffic_digest: Path | None = None,
        workload_summary: Path | None = None,
    ) -> dict[str, object]:
        from pkg.proverdet.verdict import emit_verdict

        return emit_verdict(
            self.transcript,
            traffic_digest_path=traffic_digest,
            workload_summary_path=workload_summary,
        )

    # ---- honest baseline -----------------------------------------------

    def test_honest_inference_passes_all_signals(self) -> None:
        # Two passing replays, claimed >= observed, traffic exactly equal
        # to the claimed bytes. Combiner must emit `inference`.
        _write_jsonl(
            self.transcript,
            [
                _verdict_entry(1, "r-1", status=200),
                _verdict_entry(2, "r-2", status=200),
            ],
        )
        _write_jsonl(
            self.summaries,
            [
                {"kind": "graph", "claimed_flops_total": 5120, "task_count": 2},
                {
                    "kind": "replay_evidence",
                    "replay_id": "r-1",
                    "observed_flops": 1024,
                    "rounds": 1,
                    "matmul_dim": 8,
                },
            ],
        )
        traffic_digest = self.dir / "traffic.digest"
        traffic_digest.write_text("sha256:" + "0" * 64 + "\n", encoding="utf-8")
        (self.dir / "traffic.bin").write_bytes(b"x" * 5120)

        result = self._emit(traffic_digest=traffic_digest)
        self.assertEqual(result["verdict"], "inference", result)
        self.assertEqual(result["reasons"], [])

    # ---- replay correctness fires -------------------------------------

    def test_failing_replay_status_triggers_training_or_exfil(self) -> None:
        _write_jsonl(
            self.transcript,
            [
                _verdict_entry(1, "r-1", status=200),
                _verdict_entry(2, "r-bad", status=422),
            ],
        )
        result = self._emit()
        self.assertEqual(result["verdict"], "training_or_exfil")
        reasons = result["reasons"]
        self.assertIsInstance(reasons, list)
        assert isinstance(reasons, list)
        self.assertTrue(any("r-bad" in r for r in reasons), reasons)

    # ---- compute budget fires (mixed_lora-shape) ----------------------

    def test_workload_summary_compute_budget_triggers_training(self) -> None:
        # mixed_lora's fingerprint: workload-internal observed FLOPs blow
        # past the inference budget. Without workload_summary the verdict
        # would (incorrectly) say `inference` because /graph claims 0 and
        # the scheduler-replays show 0. Threading the workload summary in
        # is the whole point of Phase 8.3's compute_budget extension.
        _write_jsonl(
            self.summaries,
            [{"kind": "graph", "claimed_flops_total": 0, "task_count": 0}],
        )
        ws_path = self.dir / "workload_summary.json"
        ws_path.write_text(
            json.dumps(
                {
                    "claimed_flops_total": 100,
                    "observed_flops_total": 10_000,
                    "task_count": 1,
                }
            ),
            encoding="utf-8",
        )
        result = self._emit(workload_summary=ws_path)
        self.assertEqual(result["verdict"], "training_or_exfil")
        reasons = result["reasons"]
        assert isinstance(reasons, list)
        self.assertTrue(any("compute" in r.lower() for r in reasons), reasons)

    # ---- bandwidth signal fires (lora_loading-shape) ------------------

    def test_bandwidth_signal_triggers_training_or_exfil(self) -> None:
        # lora_loading fingerprint: claimed bytes are tiny, traffic is huge.
        _write_jsonl(
            self.summaries,
            [{"kind": "graph", "claimed_flops_total": 256, "task_count": 1}],
        )
        traffic_digest = self.dir / "traffic.digest"
        traffic_digest.write_text("sha256:" + "0" * 64 + "\n", encoding="utf-8")
        # 100x baseline — bandwidth signal fires at zero tolerance the moment
        # traffic exceeds the claim by even one byte; 100x is unambiguous.
        (self.dir / "traffic.bin").write_bytes(b"x" * 25_600)

        result = self._emit(traffic_digest=traffic_digest)
        self.assertEqual(result["verdict"], "training_or_exfil")
        reasons = result["reasons"]
        assert isinstance(reasons, list)
        self.assertTrue(any("bandwidth" in r.lower() for r in reasons), reasons)

    # ---- multiple signals fail -> all reasons concatenated ------------

    def test_multiple_failing_signals_concatenate_reasons(self) -> None:
        # Replay fail + budget fail; both reasons should appear.
        _write_jsonl(
            self.transcript,
            [_verdict_entry(1, "r-bad", status=422)],
        )
        _write_jsonl(
            self.summaries,
            [
                {"kind": "graph", "claimed_flops_total": 100, "task_count": 1},
                {
                    "kind": "replay_evidence",
                    "replay_id": "r-bad",
                    "observed_flops": 10_000,
                    "rounds": 1,
                    "matmul_dim": 8,
                },
            ],
        )
        result = self._emit()
        self.assertEqual(result["verdict"], "training_or_exfil")
        reasons = result["reasons"]
        assert isinstance(reasons, list)
        self.assertGreaterEqual(len(reasons), 2, reasons)
        joined = " ".join(reasons).lower()
        self.assertIn("r-bad", joined)
        self.assertIn("compute", joined)


if __name__ == "__main__":
    unittest.main()
