"""Concrete WorkloadContext shared by every workload.

This is *not* an ABC. The first concrete workload (`benign.py`) defines the
duck-typed shape it expects from a `ctx`; the second and third workloads
inherit nothing — they just call the same methods. If a fourth workload
ever needs something extra, add the field here, not in a Protocol.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class WorkloadContext:
    """Shared state passed to every workload's run loop."""

    # Network: every byte the workload puts on the wire goes through this
    # function. Tests pass a recorder; production passes
    # TrafficPublisher.publish.
    publish_frame: Callable[[bytes], None]

    # Graph: workload appends one record per claimed task. The prover
    # exposes the accumulated list via /graph in Phase 6+.
    record_task: Callable[[dict[str, object]], None]

    # Cooperative cancellation. The prover's /workload/stop sets this.
    stop_event: threading.Event = field(default_factory=threading.Event)
