"""Shared base for the prover capture log and the verifier transcript log.

Both are append-only JSONL logs of (direction, endpoint, payload_digest)
events with a monotonic seq, threadsafe across handler threads. The
verifier transcript layers in schema-validation per append; the prover
capture log doesn't (it's not consumed by the verdict engine).

Subclasses set SCHEMA_NAME (or leave as None to skip validation) and
otherwise reuse the threadsafe append + canonical-JSON write here.
"""

from __future__ import annotations

import threading
from pathlib import Path

from modules.core.common.contracts import validate_with_schema
from modules.core.common.deterministic import (
    canonical_json_text,
    sha256_prefixed,
    utc_now_iso,
)


class JsonlLog:
    """Threadsafe append-only JSONL log with monotonic seq."""

    SCHEMA_NAME: str | None = None

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()
        self._seq = 0

    def record(
        self,
        *,
        direction: str,
        endpoint: str,
        payload: bytes,
        status_code: int | None = None,
        payload_path: str | None = None,
    ) -> int:
        with self._lock:
            self._seq += 1
            seq = self._seq
            entry: dict[str, object] = {
                "seq": seq,
                "direction": direction,
                "endpoint": endpoint,
                "timestamp": utc_now_iso(),
                "payload_digest": sha256_prefixed(payload),
            }
            if status_code is not None:
                entry["status_code"] = status_code
            if payload_path is not None:
                entry["payload_path"] = payload_path

            if self.SCHEMA_NAME is not None:
                validate_with_schema(self.SCHEMA_NAME, entry)

            with self.path.open("a", encoding="utf-8") as f:
                f.write(canonical_json_text(entry))
            return seq

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq
