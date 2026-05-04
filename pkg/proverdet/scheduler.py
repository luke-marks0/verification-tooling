"""Verifier scheduler — the active half of the verifier.

Polls the prover with /graph requests at one cadence and /replay challenges
at another. Records every send/recv to the transcript. Reproducible
across runs given the same seed.

For tests, inject a fake client + fake clock; for production, use
HttpProverClient + WallClock. The scheduler doesn't know the difference.
"""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from pkg.common.deterministic import canonical_json_text
from pkg.proverdet.transcript import TranscriptLog


class ProverClient(Protocol):
    def get_graph(self) -> tuple[int, dict[str, object]]: ...
    def post_replay(self, request: dict[str, object]) -> tuple[int, dict[str, object]]: ...


class Clock(Protocol):
    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


@dataclass
class WallClock:
    """Real-clock implementation of Clock."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def _build_replay_request(replay_id: str, rng: random.Random) -> dict[str, object]:
    # Stub-targets only at this phase (3.3); Phase 6+ will pick real
    # task/artifact ids based on the most recent /graph response.
    return {
        "replay_id": replay_id,
        "pod_id": "stub-pod",
        "target": {"kind": "task", "task_id": "stub-task-0"},
        "erasure": {
            "challenge_seed": f"{rng.getrandbits(64):016x}",
            "deadline_ms": 1000,
            "rounds": 2,
        },
        "proof_of_work": {
            "matmul_dim": 8,
            "dtype": "int8",
            "rounds": 1,
            "report_every_ms": 100,
        },
        "auxiliary": [],
    }


class VerifierScheduler:
    """Active scheduler issuing /graph and /replay against a prover.

    Tick-based. `run_for_ticks(n)` advances `n` discrete steps of size
    `tick_ms`. `start()` runs in a daemon thread until `stop()` is called.
    The behaviour is the same — they share the same loop body — but tests
    use `run_for_ticks` for full determinism.
    """

    def __init__(
        self,
        *,
        client: ProverClient,
        transcript: TranscriptLog,
        seed: int = 0,
        tick_ms: int = 50,
        graph_period_ms: int = 1000,
        replay_period_ms: int = 2000,
        clock: Clock | None = None,
    ) -> None:
        self.client = client
        self.transcript = transcript
        self._rng = random.Random(seed)
        self.tick_ms = tick_ms
        self.graph_period_ms = graph_period_ms
        self.replay_period_ms = replay_period_ms
        self.clock = clock or WallClock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_graph_due = 0
        self._next_replay_due = 0
        self._tick = 0
        self._replay_counter = 0

    def _tick_once(self) -> None:
        if self._tick * self.tick_ms >= self._next_graph_due:
            self._do_graph()
            self._next_graph_due = (self._tick + 1) * self.tick_ms + self.graph_period_ms
        if self._tick * self.tick_ms >= self._next_replay_due:
            self._do_replay()
            self._next_replay_due = (self._tick + 1) * self.tick_ms + self.replay_period_ms
        self._tick += 1

    def run_for_ticks(self, n: int) -> None:
        for _ in range(n):
            self._tick_once()
            self.clock.sleep(self.tick_ms / 1000.0)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("scheduler already started")
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                self._tick_once()
                # Real WallClock; daemon thread can yield via short sleep.
                self.clock.sleep(self.tick_ms / 1000.0)

        self._thread = threading.Thread(target=loop, daemon=True, name="VerifierScheduler")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # -- internal --

    def _record_sent(self, endpoint: str, payload: bytes) -> None:
        self.transcript.record(direction="sent", endpoint=endpoint, payload=payload)

    def _record_received(self, endpoint: str, payload: bytes, status: int) -> None:
        self.transcript.record(
            direction="received",
            endpoint=endpoint,
            payload=payload,
            status_code=status,
        )

    def _do_graph(self) -> None:
        # Sent payload is empty for GET /graph; we still record so the
        # transcript shows the request happened.
        self._record_sent("/graph", b"")
        try:
            status, body = self.client.get_graph()
        except Exception as exc:
            self._record_received("/graph", str(exc).encode("utf-8"), 599)
            return
        body_bytes = canonical_json_text(body).encode("utf-8")
        self._record_received("/graph", body_bytes, status)

    def _do_replay(self) -> None:
        self._replay_counter += 1
        replay_id = f"r-{self._replay_counter:06d}-{self._rng.getrandbits(32):08x}"
        request = _build_replay_request(replay_id, self._rng)
        request_bytes = canonical_json_text(request).encode("utf-8")
        self._record_sent("/replay", request_bytes)
        try:
            status, body = self.client.post_replay(request)
        except Exception as exc:
            self._record_received("/replay", str(exc).encode("utf-8"), 599)
            return
        body_bytes = canonical_json_text(body).encode("utf-8")
        self._record_received("/replay", body_bytes, status)


# -- HTTP client used in production (verifier server's daemon thread) --


class HttpProverClient:
    """Talk to the prover over HTTP using stdlib urllib."""

    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def get_graph(self) -> tuple[int, dict[str, object]]:
        import urllib.request

        with urllib.request.urlopen(f"{self.base_url}/graph", timeout=self.timeout_s) as r:
            body = r.read()
            return r.status, json.loads(body) if body else {}

    def post_replay(self, request: dict[str, object]) -> tuple[int, dict[str, object]]:
        import urllib.error
        import urllib.request

        data = canonical_json_text(request).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/replay",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                body = r.read()
                return r.status, json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            body = e.read()
            return e.code, json.loads(body) if body else {}
