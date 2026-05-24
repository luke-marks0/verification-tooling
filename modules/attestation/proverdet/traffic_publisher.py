"""Prover traffic publisher.

The prover's DeterministicNetStack emits frames; this class plumbs those
bytes to the verifier's /traffic endpoint as batched POSTs. We pick
batched POSTs (concatenate up to N bytes per request) over a long-lived
chunked stream because Python stdlib chunked-body uploads have rough
edges across versions and the verifier sees the same concatenated bytes
either way. (Plan §4.1.)
"""

from __future__ import annotations

import queue
import threading
import urllib.error
import urllib.request

_DEFAULT_BATCH_BYTES = 64 * 1024  # 64 KiB
_FLUSH_INTERVAL_S = 0.05  # 50 ms — drain even if batch is half-full
_DRAIN_SENTINEL: object = object()


class TrafficPublisher:
    """Background-thread publisher that POSTs batched frames to the verifier."""

    def __init__(
        self,
        *,
        verifier_url: str,
        max_batch_bytes: int = _DEFAULT_BATCH_BYTES,
        flush_interval_s: float = _FLUSH_INTERVAL_S,
        timeout_s: float = 5.0,
    ) -> None:
        self.verifier_url = verifier_url.rstrip("/")
        self.max_batch_bytes = max_batch_bytes
        self.flush_interval_s = flush_interval_s
        self.timeout_s = timeout_s
        # Unbounded — workload threads should never block on publish.
        self._queue: queue.Queue[bytes | object] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(target=self._run, name="TrafficPublisher", daemon=True)
            self._thread.start()

    def publish(self, frame: bytes) -> None:
        if not frame:
            return
        self._queue.put(frame)

    def stop(self, timeout: float = 10.0) -> None:
        """Drain remaining frames and join the worker thread."""
        with self._lock:
            if not self._started:
                return
            self._queue.put(_DRAIN_SENTINEL)
            t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        with self._lock:
            self._started = False
            self._thread = None

    # -- internals --

    def _run(self) -> None:
        buf = bytearray()
        while True:
            try:
                item = self._queue.get(timeout=self.flush_interval_s)
            except queue.Empty:
                if buf:
                    self._flush(bytes(buf))
                    buf = bytearray()
                continue

            if item is _DRAIN_SENTINEL:
                # Drain everything queued before the sentinel, plus the
                # in-flight buffer.
                while True:
                    try:
                        more = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if more is _DRAIN_SENTINEL:
                        continue
                    assert isinstance(more, bytes)
                    buf.extend(more)
                if buf:
                    self._flush(bytes(buf))
                return

            assert isinstance(item, bytes)
            buf.extend(item)
            if len(buf) >= self.max_batch_bytes:
                self._flush(bytes(buf))
                buf = bytearray()

    def _flush(self, data: bytes) -> None:
        req = urllib.request.Request(
            f"{self.verifier_url}/traffic",
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                r.read()
        except urllib.error.URLError:
            # Demo-time: dropping a batch on a transient error is cheap;
            # recovery semantics are deferred (see plan §4.1 "out of scope
            # for the demo").
            return
