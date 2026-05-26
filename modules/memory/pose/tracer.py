"""Simple flat trace collector for benchmarking."""

import time
from contextlib import contextmanager


class Tracer:
    def __init__(self):
        self._events = []

    @contextmanager
    def step(self, name: str, bytes: int | None = None):
        start = time.monotonic()
        start_wall = time.time()
        event = {
            "step": name,
            "start_ts": start_wall,
            "end_ts": None,
            "bytes": bytes,
        }
        self._events.append(event)
        yield
        end = time.monotonic()
        event["end_ts"] = start_wall + (end - start)

    def events(self) -> list[dict]:
        return list(self._events)
