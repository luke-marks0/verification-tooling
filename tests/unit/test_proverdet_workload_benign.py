from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

EXP_SCRIPTS = (
    Path(__file__).resolve().parents[2] / "experiments" / "prover-verifier-demo" / "scripts"
)
if str(EXP_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EXP_SCRIPTS))

from workloads.benign import BenignInferenceWorkload  # noqa: E402
from workloads.context import WorkloadContext  # noqa: E402


def _make_recording_ctx() -> tuple[WorkloadContext, list[bytes], list[dict]]:
    frames: list[bytes] = []
    tasks: list[dict] = []
    ctx = WorkloadContext(
        publish_frame=frames.append,
        record_task=tasks.append,
        stop_event=threading.Event(),
    )
    return ctx, frames, tasks


class TestBenignWorkloadCpu(unittest.TestCase):
    def test_synthetic_path_publishes_one_frame_set_per_prompt(self) -> None:
        ctx, frames, _tasks = _make_recording_ctx()
        wl = BenignInferenceWorkload(prompts=["hello", "world", "!"], use_vllm=False, seed=7)
        wl.run(ctx)
        # 10 frames per prompt, 256 bytes each.
        self.assertEqual(len(frames), 30)
        for f in frames:
            self.assertEqual(len(f), 256)

    def test_synthetic_path_records_one_task_per_prompt(self) -> None:
        ctx, _frames, tasks = _make_recording_ctx()
        wl = BenignInferenceWorkload(prompts=["a", "b", "c"], use_vllm=False, seed=7)
        wl.run(ctx)
        self.assertEqual(len(tasks), 3)
        for t in tasks:
            self.assertEqual(t["pod_id"], "pod-a")
            self.assertEqual(t["operation"], "inference")
            self.assertGreater(t["claimed_flops"], 0)

    def test_synthetic_path_is_deterministic_for_a_seed(self) -> None:
        ctx_a, fr_a, _ = _make_recording_ctx()
        ctx_b, fr_b, _ = _make_recording_ctx()
        BenignInferenceWorkload(prompts=["x", "y"], use_vllm=False, seed=7).run(ctx_a)
        BenignInferenceWorkload(prompts=["x", "y"], use_vllm=False, seed=7).run(ctx_b)
        self.assertEqual(fr_a, fr_b)

    def test_synthetic_path_obeys_stop_event(self) -> None:
        ctx, frames, tasks = _make_recording_ctx()
        ctx.stop_event.set()
        wl = BenignInferenceWorkload(prompts=["hello", "world"], use_vllm=False, seed=7)
        wl.run(ctx)
        # Stopped immediately — no work done.
        self.assertEqual(frames, [])
        self.assertEqual(tasks, [])


if __name__ == "__main__":
    unittest.main()
