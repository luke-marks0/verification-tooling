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

from workloads.context import WorkloadContext  # noqa: E402
from workloads.mixed_lora import MixedLoraWorkload  # noqa: E402


def _make_recording_ctx() -> tuple[WorkloadContext, list[bytes], list[dict]]:
    frames: list[bytes] = []
    tasks: list[dict] = []
    ctx = WorkloadContext(
        publish_frame=frames.append,
        record_task=tasks.append,
        stop_event=threading.Event(),
    )
    return ctx, frames, tasks


class TestMixedLoraWorkloadCpu(unittest.TestCase):
    def test_records_only_inference_tasks(self) -> None:
        ctx, _frames, tasks = _make_recording_ctx()
        wl = MixedLoraWorkload(
            prompts=["a", "b", "c"],
            gradient_steps=2,
            matmul_dim=64,
            seed=7,
        )
        wl.run(ctx)
        # One task per inference prompt; gradient steps DO NOT emit tasks.
        self.assertEqual(len(tasks), 3)
        for t in tasks:
            self.assertEqual(t["operation"], "inference")

    def test_publishes_only_inference_frames(self) -> None:
        ctx, frames, _tasks = _make_recording_ctx()
        wl = MixedLoraWorkload(
            prompts=["a", "b"],
            gradient_steps=4,
            matmul_dim=64,
            seed=7,
        )
        wl.run(ctx)
        # 10 frames per inference prompt (mirrors benign workload).
        self.assertEqual(len(frames), 20)

    def test_observed_flops_exceeds_claimed_by_gradient_compute(self) -> None:
        ctx, _frames, tasks = _make_recording_ctx()
        inference_count = 3
        gradient_steps = 2
        dim = 64
        wl = MixedLoraWorkload(
            prompts=[f"p{i}" for i in range(inference_count)],
            gradient_steps=gradient_steps,
            matmul_dim=dim,
            seed=7,
        )
        wl.run(ctx)

        claimed = sum(t["claimed_flops"] for t in tasks)
        gap = wl.observed_flops_total - claimed
        # Each gradient step is one matmul of dim^3 multiplies + dim^3 adds.
        # Tolerate small modeling differences but require the headline gap.
        expected_gap = inference_count * gradient_steps * 2 * dim**3
        self.assertGreater(wl.observed_flops_total, claimed)
        self.assertEqual(gap, expected_gap)

    def test_zero_gradient_steps_matches_benign_accounting(self) -> None:
        ctx, _frames, tasks = _make_recording_ctx()
        wl = MixedLoraWorkload(
            prompts=["a", "b"],
            gradient_steps=0,
            matmul_dim=64,
            seed=7,
        )
        wl.run(ctx)
        claimed = sum(t["claimed_flops"] for t in tasks)
        # No cheating ⇒ observed == claimed.
        self.assertEqual(wl.observed_flops_total, claimed)

    def test_synthetic_path_is_deterministic_for_a_seed(self) -> None:
        ctx_a, fr_a, _ = _make_recording_ctx()
        ctx_b, fr_b, _ = _make_recording_ctx()
        MixedLoraWorkload(prompts=["x", "y"], gradient_steps=2, seed=7).run(ctx_a)
        MixedLoraWorkload(prompts=["x", "y"], gradient_steps=2, seed=7).run(ctx_b)
        self.assertEqual(fr_a, fr_b)

    def test_obeys_stop_event(self) -> None:
        ctx, frames, tasks = _make_recording_ctx()
        ctx.stop_event.set()
        wl = MixedLoraWorkload(prompts=["a", "b"], gradient_steps=2, seed=7)
        wl.run(ctx)
        self.assertEqual(frames, [])
        self.assertEqual(tasks, [])


if __name__ == "__main__":
    unittest.main()
