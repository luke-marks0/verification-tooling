from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

# Workloads live under experiments/prover-verifier-demo/scripts/workloads;
# add the experiment scripts dir to sys.path so the test can import them.
EXP_SCRIPTS = (
    Path(__file__).resolve().parents[2] / "experiments" / "prover-verifier-demo" / "scripts"
)
if str(EXP_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EXP_SCRIPTS))

from workloads.context import WorkloadContext  # noqa: E402


class TestWorkloadContext(unittest.TestCase):
    def test_publish_frame_is_called(self) -> None:
        published: list[bytes] = []

        def publish(b: bytes) -> None:
            published.append(b)

        ctx = WorkloadContext(publish_frame=publish, record_task=lambda r: None)
        ctx.publish_frame(b"hello")
        ctx.publish_frame(b"world")
        self.assertEqual(published, [b"hello", b"world"])

    def test_record_task_is_called(self) -> None:
        recorded: list[dict[str, object]] = []
        ctx = WorkloadContext(
            publish_frame=lambda b: None,
            record_task=recorded.append,
        )
        ctx.record_task({"task_id": "t-0"})
        self.assertEqual(recorded, [{"task_id": "t-0"}])

    def test_stop_event_default_is_clear(self) -> None:
        ctx = WorkloadContext(
            publish_frame=lambda b: None,
            record_task=lambda r: None,
        )
        self.assertFalse(ctx.stop_event.is_set())

    def test_setting_stop_event_does_not_affect_routing(self) -> None:
        # The context itself doesn't observe stop_event; workloads do.
        published: list[bytes] = []
        ctx = WorkloadContext(
            publish_frame=published.append,
            record_task=lambda r: None,
        )
        ctx.stop_event.set()
        ctx.publish_frame(b"after stop")
        self.assertEqual(published, [b"after stop"])

    def test_stop_event_is_a_threading_event(self) -> None:
        ctx = WorkloadContext(
            publish_frame=lambda b: None,
            record_task=lambda r: None,
        )
        self.assertIsInstance(ctx.stop_event, threading.Event)


if __name__ == "__main__":
    unittest.main()
