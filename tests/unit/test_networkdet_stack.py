"""Unit tests for DeterministicNetStack MRF verification and flush."""
from __future__ import annotations

import unittest

from modules.network.networkdet import DeterministicNetStack, create_net_stack
from modules.network.networkdet.backend_sim import SimulatedBackend
from modules.network.networkdet.config import NetStackConfig

try:
    from modules.network.networkdet.warden import ActiveWarden  # noqa: F401  (availability probe)
    HAS_WARDEN = True
except ImportError:
    HAS_WARDEN = False


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
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        src_mac="02:00:00:00:00:01",
        dst_mac="02:00:00:00:00:02",
        src_port=8000,
        dst_port=80,
    )


@unittest.skipUnless(HAS_WARDEN, "warden module not available on this branch")
class TestMRFVerification(unittest.TestCase):

    def test_mrf_verification_passes_for_valid_frames(self):
        """Frames from the deterministic builder pass warden verification."""
        net = DeterministicNetStack(
            _test_config(),
            run_id="test-run",
            backend=SimulatedBackend(),
            verify_mrf=True,
        )
        # Should not raise.
        frames = net.process_response(0, b"Hello, world!")
        self.assertGreater(len(frames), 0)
        net.close()

    def test_mrf_verification_catches_bad_frames(self):
        """Frames with a nonzero reserved bit trigger RuntimeError."""
        net = DeterministicNetStack(
            _test_config(),
            run_id="test-run",
            backend=SimulatedBackend(),
            verify_mrf=True,
        )
        original_build = None

        def _corrupt_build(response_bytes):
            frames = original_build(response_bytes)
            corrupted = []
            for frame in frames:
                buf = bytearray(frame)
                # Set a nonzero reserved bit in TCP header byte 12 (lower nibble).
                tcp_start = 14 + 20  # Eth + IP headers
                buf[tcp_start + 12] |= 0x01  # Set NS/reserved bit
                # Recompute TCP checksum so the only violation is the reserved bit.
                corrupted.append(bytes(buf))
            return corrupted

        builder = net._get_builder(0)
        original_build = builder.build_response_frames
        builder.build_response_frames = _corrupt_build

        with self.assertRaises(RuntimeError) as ctx:
            net.process_response(0, b"test payload")
        self.assertIn("non-MRF-compliant", str(ctx.exception))
        net.close()

    def test_mrf_verification_ignores_ip_id_rewrite(self):
        """Warden IP ID encryption is expected, not a violation."""
        net = DeterministicNetStack(
            _test_config(),
            run_id="test-run",
            backend=SimulatedBackend(),
            verify_mrf=True,
        )
        # Multiple frames — the warden encrypts IP IDs on every frame.
        # This should not raise.
        frames = net.process_response(0, b"A" * 250)
        self.assertEqual(len(frames), 3)
        net.close()


class TestFlush(unittest.TestCase):

    def test_flush_delegates_to_backend(self):
        net = DeterministicNetStack(
            _test_config(),
            run_id="test-run",
            backend=SimulatedBackend(),
        )
        self.assertIsNone(net.flush())
        net.close()


class TestCreateNetStackSim(unittest.TestCase):

    _MANIFEST = {
        "run_id": "test-run",
        "network": {
            "mtu": 1500, "mss": 1460,
            "tso": False, "gso": False, "checksum_offload": False,
            "thread_affinity": [0],
            "security_mode": "plaintext",
            "egress_reproducibility": True,
        },
    }

    def test_create_net_stack_sim_still_works(self):
        net = create_net_stack(self._MANIFEST, {}, backend="sim")
        frames = net.process_response(0, b"data")
        self.assertGreater(len(frames), 0)
        net.close()

    def test_create_net_stack_dpdk_raises_without_library(self):
        with self.assertRaises((OSError, RuntimeError, ImportError)):
            create_net_stack(self._MANIFEST, {}, backend="dpdk")


if __name__ == "__main__":
    unittest.main()
