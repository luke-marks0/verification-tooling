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
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from pkg.common.deterministic import canonical_json_text
from pkg.freivalds.backends.stdlib import StdlibBackend
from pkg.proverdet.replay import FreivaldsBackend
from pkg.proverdet.replay_verify import verify_evidence
from pkg.proverdet.transcript import TranscriptLog
from pkg.proverdet.wire import ReplayEvidence, ReplayRequest


class ProverClient(Protocol):
    def get_graph(self) -> tuple[int, dict[str, object]]: ...
    def post_replay(
        self, request: dict[str, object]
    ) -> Iterator[tuple[int, dict[str, object]]]: ...
    def get_attestation(self, attestation_id: str) -> dict[str, object] | None: ...


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
        backend: FreivaldsBackend | None = None,
    ) -> None:
        self.client = client
        self.transcript = transcript
        self._rng = random.Random(seed)
        self.tick_ms = tick_ms
        self.graph_period_ms = graph_period_ms
        self.replay_period_ms = replay_period_ms
        self.clock = clock or WallClock()
        # The backend is what verify_evidence runs Freivalds with on each
        # received pow attestation. Stdlib is enough on CPU; tests inject
        # the same.
        self.backend: FreivaldsBackend = backend or StdlibBackend()
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
            stream = self.client.post_replay(request)
        except Exception as exc:
            self._record_received("/replay", str(exc).encode("utf-8"), 599)
            return
        # Each NDJSON chunk -> one transcript entry. The final chunk
        # (kind=evidence) lands as the last received entry; intermediate
        # `kind=pow` chunks land in arrival order. Errors mid-stream are
        # captured as a `kind=error` chunk by the prover; we simply record
        # them like any other.
        evidence_chunk: dict[str, object] | None = None
        try:
            for status, chunk in stream:
                chunk_bytes = canonical_json_text(chunk).encode("utf-8")
                self._record_received("/replay", chunk_bytes, status)
                if chunk.get("kind") == "evidence":
                    evidence_chunk = chunk
        except Exception as exc:
            self._record_received("/replay", str(exc).encode("utf-8"), 599)
            return

        if evidence_chunk is None:
            return
        # Verify the evidence and append a verdict transcript entry.
        try:
            req = ReplayRequest.model_validate(request)
            ev_body = {k: v for k, v in evidence_chunk.items() if k != "kind"}
            ev = ReplayEvidence.model_validate(ev_body)
        except Exception as exc:
            self._record_received("/replay/verdict", f"parse error: {exc}".encode(), 599)
            return
        verdict = verify_evidence(
            req,
            ev,
            fetch_attestation=self.client.get_attestation,
            backend=self.backend,
        )
        verdict_bytes = canonical_json_text(
            {"replay_id": ev.replay_id, "passed": verdict.passed, "reasons": verdict.reasons}
        ).encode("utf-8")
        # Embed the replay_id in the endpoint so signals (Phase 8) can
        # name failing replays without depending on the payload bytes —
        # the transcript only records the digest.
        self._record_received(
            f"/replay/verdict/{ev.replay_id}",
            verdict_bytes,
            200 if verdict.passed else 422,
        )


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

    def get_attestation(self, attestation_id: str) -> dict[str, object] | None:
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(
                f"{self.base_url}/attestation/{attestation_id}", timeout=self.timeout_s
            ) as r:
                body = r.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def post_replay(self, request: dict[str, object]) -> Iterator[tuple[int, dict[str, object]]]:
        """Stream the prover's NDJSON /replay response chunk by chunk."""
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
                status = r.status
                for raw_line in r:
                    line = raw_line.strip()
                    if not line:
                        continue
                    yield status, json.loads(line)
        except urllib.error.HTTPError as e:
            body = e.read()
            payload: dict[str, object] = {}
            if body:
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    payload = {"error": body.decode("utf-8", errors="replace")}
            yield e.code, payload
