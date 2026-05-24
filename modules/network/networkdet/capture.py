"""Non-perturbing capture ring (SPEC-9.3).

The capture ring receives copies of L2 frames before they are enqueued
for transmission.  Recording a frame does not alter the frame or affect
the order/timing of transmission — the frame bytes are copied, not moved.

This satisfies the spec requirement that "capturing network egress MUST
NOT affect packetization or ordering".
"""
from __future__ import annotations

import hashlib


class CaptureRing:
    """Pre-enqueue mirror buffer for deterministic frame capture."""

    def __init__(self) -> None:
        self._frames: list[bytes] = []

    def record(self, frame: bytes) -> None:
        """Record a copy of a frame before transmission.

        The frame bytes are copied into the ring.  The original frame
        object is not modified or retained.
        """
        self._frames.append(bytes(frame))

    def drain(self) -> list[bytes]:
        """Return all captured frames and clear the ring."""
        frames = list(self._frames)
        self._frames.clear()
        return frames

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def digest(self) -> str:
        """Compute SHA-256 digest over the ordered frame concatenation.

        Returns a prefixed digest string: ``sha256:<hex>``.
        """
        h = hashlib.sha256()
        for frame in self._frames:
            h.update(frame)
        return f"sha256:{h.hexdigest()}"

    def frames_as_hex(self) -> list[dict[str, object]]:
        """Return frames as a list of dicts suitable for JSON serialization."""
        return [
            {"frame_index": i, "frame_hex": frame.hex()}
            for i, frame in enumerate(self._frames)
        ]
