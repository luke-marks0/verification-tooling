"""In-memory attestation store.

Each /replay produces one Freivalds attestation per round. We stash the
attestation body (challenge + response + matmul_id) under an opaque id and
expose it via GET /attestation/{id}. The verifier fetches stored
attestations to re-run Freivalds checks server-side without round-tripping
the matrix bytes inside /replay's response itself.

Lifetimes match the prover process; nothing is persisted to disk.
"""

from __future__ import annotations

import threading


class AttestationStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, dict[str, object]] = {}

    def put(self, attestation_id: str, body: dict[str, object]) -> None:
        with self._lock:
            self._items[attestation_id] = body

    def get(self, attestation_id: str) -> dict[str, object] | None:
        with self._lock:
            return self._items.get(attestation_id)
