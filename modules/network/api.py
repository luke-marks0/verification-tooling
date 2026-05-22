"""Deterministic network egress — stable public API. See ``README.md``.

Turns application-layer bytes into bitwise-reproducible L2 frames (a real NIC
via DPDK, or a pure-Python ``sim`` backend for tests). Re-exports the
``pkg/networkdet`` primitives as the curated public surface.
"""
from __future__ import annotations

from typing import Any

from pkg.networkdet import DeterministicNetStack, create_net_stack

__all__ = ["create_net_stack", "DeterministicNetStack", "egress_frames"]


def egress_frames(
    payload: bytes,
    *,
    manifest: dict[str, Any],
    lockfile: dict[str, Any],
    dst_mac: str = "02:00:00:00:00:02",
    backend: str = "sim",
    conn_index: int = 0,
) -> list[bytes]:
    """Turn one application-layer payload into deterministic L2 frames.

    The headline "send data, get deterministic egress" entry point: two calls
    with the same payload and config produce byte-identical frames.

    Args:
        payload: Application-layer response bytes.
        manifest/lockfile: Parsed spine artifacts (drive the network config).
        dst_mac: Destination MAC for the egress frames.
        backend: ``"sim"`` (pure Python) or ``"dpdk"`` (real NIC).
        conn_index: Logical connection index (deterministic per-connection ISN).
    """
    stack = create_net_stack(manifest, lockfile, backend=backend, dst_mac=dst_mac)
    try:
        return stack.process_response(conn_index, payload)
    finally:
        stack.close()
