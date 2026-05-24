"""Parse manifest network configuration into a validated NetStackConfig."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class NetConfigError(Exception):
    """Raised when the manifest network section is invalid for determinism."""


@dataclass(frozen=True)
class NetStackConfig:
    """Immutable network stack configuration derived from a manifest."""

    mtu: int
    mss: int
    tso: bool
    gso: bool
    checksum_offload: bool
    thread_affinity: tuple[int, ...]
    tx_queues: int
    rx_queues: int
    queue_mapping_policy: str
    ring_tx: int
    ring_rx: int
    internal_batching_enabled: bool
    internal_batching_max_burst: int
    security_mode: str
    egress_reproducibility: bool

    # Addressing — populated at stack init, not from manifest.
    src_ip: str = "0.0.0.0"
    dst_ip: str = "0.0.0.0"
    src_mac: str = "00:00:00:00:00:00"
    dst_mac: str = "00:00:00:00:00:00"
    src_port: int = 8000
    dst_port: int = 0


def parse_net_config(manifest: dict[str, Any]) -> NetStackConfig:
    """Parse and validate the ``network`` section of a manifest.

    Raises :class:`NetConfigError` if the configuration is incompatible
    with deterministic operation.
    """
    net = manifest.get("network")
    if net is None:
        # Return sensible defaults when the network section is absent.
        return NetStackConfig(
            mtu=1500, mss=1460, tso=False, gso=False,
            checksum_offload=False, thread_affinity=(0,),
            tx_queues=1, rx_queues=1,
            queue_mapping_policy="fixed_core_queue",
            ring_tx=512, ring_rx=512,
            internal_batching_enabled=False, internal_batching_max_burst=1,
            security_mode="plaintext", egress_reproducibility=True,
        )

    tso = bool(net.get("tso", False))
    gso = bool(net.get("gso", False))
    checksum_offload = bool(net.get("checksum_offload", False))

    if tso:
        raise NetConfigError(
            "TSO must be disabled for deterministic networking: "
            "hardware segmentation offload introduces NIC-firmware-dependent nondeterminism"
        )
    if gso:
        raise NetConfigError(
            "GSO must be disabled for deterministic networking: "
            "generic segmentation offload introduces kernel-dependent nondeterminism"
        )
    if checksum_offload:
        raise NetConfigError(
            "Checksum offload must be disabled for deterministic networking: "
            "hardware checksum computation is not guaranteed to be bitwise-identical"
        )

    queue_mapping = net.get("queue_mapping", {})
    internal_batching = net.get("internal_batching", {})

    return NetStackConfig(
        mtu=int(net.get("mtu", 1500)),
        mss=int(net.get("mss", 1460)),
        tso=tso,
        gso=gso,
        checksum_offload=checksum_offload,
        thread_affinity=tuple(net.get("thread_affinity", [0])),
        tx_queues=int(queue_mapping.get("tx_queues", 1)),
        rx_queues=int(queue_mapping.get("rx_queues", 1)),
        queue_mapping_policy=str(queue_mapping.get("mapping_policy", "fixed_core_queue")),
        ring_tx=int(net.get("ring_sizes", {}).get("tx", 512)),
        ring_rx=int(net.get("ring_sizes", {}).get("rx", 512)),
        internal_batching_enabled=bool(internal_batching.get("enabled", False)),
        internal_batching_max_burst=int(internal_batching.get("max_burst", 1)),
        security_mode=str(net.get("security_mode", "plaintext")),
        egress_reproducibility=bool(net.get("egress_reproducibility", True)),
    )
