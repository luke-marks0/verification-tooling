"""Unit tests for SimulatedBackend."""
from __future__ import annotations

import unittest

from modules.network.networkdet.backend_sim import SimulatedBackend
from modules.network.networkdet.config import NetStackConfig


def _test_config() -> NetStackConfig:
    return NetStackConfig(
        mtu=1500,
        mss=100,
        tso=False,
        gso=False,
        checksum_offload=False,
        thread_affinity=(0,),
        tx_queues=1,
        rx_queues=1,
        queue_mapping_policy="fixed_core_queue",
        ring_tx=512,
        ring_rx=512,
        internal_batching_enabled=False,
        internal_batching_max_burst=1,
        security_mode="plaintext",
        egress_reproducibility=True,
    )


class TestSimulatedBackend(unittest.TestCase):

    def test_flush_returns_none(self):
        backend = SimulatedBackend()
        backend.init(_test_config())
        self.assertIsNone(backend.flush())


if __name__ == "__main__":
    unittest.main()
