"""Workload runner: name → class lookup + thread management.

The lookup is a literal dict — no entry points, no plugin discovery. If
you add a workload, edit `WORKLOAD_REGISTRY` here.

Workloads run in a daemon thread; start() returns immediately. stop()
sets the workload context's stop_event and joins the thread.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_SCRIPTS = REPO_ROOT / "experiments" / "prover-verifier-demo" / "scripts"


def _ensure_experiment_scripts_on_path() -> None:
    if str(EXP_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(EXP_SCRIPTS))


def _resolve_workload_class(name: str) -> type:
    """Map workload name → class. Imports lazily."""
    _ensure_experiment_scripts_on_path()

    if name == "benign":
        from workloads.benign import BenignInferenceWorkload  # type: ignore[import-not-found]

        return BenignInferenceWorkload
    if name == "mixed_lora":
        from workloads.mixed_lora import MixedLoraWorkload  # type: ignore[import-not-found]

        return MixedLoraWorkload
    if name == "lora_loading":
        from workloads.lora_loading import LoraLoadingWorkload  # type: ignore[import-not-found]

        return LoraLoadingWorkload
    raise KeyError(f"unknown workload: {name!r}")


class WorkloadRunner:
    """Owns the at-most-one currently-running workload thread."""

    def __init__(
        self,
        *,
        publish_frame: Callable[[bytes], None],
        record_task: Callable[[dict[str, object]], None],
    ) -> None:
        self.publish_frame = publish_frame
        self.record_task = record_task
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._current_name: str | None = None
        # Reference to the currently-running workload object. We don't
        # define a Protocol — we just duck-type read `observed_flops_total`
        # in stop(). Keeps adversarial workloads' bookkeeping accessible
        # to the prover server's /workload/stop response.
        self._current_workload: object | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self, *, name: str, params: dict[str, Any]) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError(f"workload already running: {self._current_name}")
            cls = _resolve_workload_class(name)
            workload = cls(**params)

            _ensure_experiment_scripts_on_path()
            from workloads.context import WorkloadContext  # type: ignore[import-not-found]

            stop_event = threading.Event()
            ctx = WorkloadContext(
                publish_frame=self.publish_frame,
                record_task=self.record_task,
                stop_event=stop_event,
            )

            def runner() -> None:
                workload.run(ctx)

            t = threading.Thread(target=runner, daemon=True, name=f"workload-{name}")
            self._thread = t
            self._stop_event = stop_event
            self._current_name = name
            self._current_workload = workload
            t.start()

    def stop(self, timeout: float = 10.0) -> int:
        """Stop the workload thread; return its observed_flops_total (or 0)."""
        with self._lock:
            ev = self._stop_event
            t = self._thread
            workload = self._current_workload
            self._stop_event = None
            self._thread = None
            self._current_name = None
            self._current_workload = None
        if ev is not None:
            ev.set()
        if t is not None:
            t.join(timeout=timeout)
        # Duck-type read: adversarial workloads expose their own internal
        # FLOPs counter. Phase 8.3's compute_budget signal compares this
        # against the graph's claimed_flops_total.
        observed = 0
        if workload is not None:
            attr = getattr(workload, "observed_flops_total", 0)
            if isinstance(attr, int):
                observed = attr
        return observed
