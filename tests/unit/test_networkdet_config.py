"""Unit tests for network configuration parsing and validation."""
from __future__ import annotations

import unittest

from modules.network.networkdet.config import NetConfigError, NetStackConfig, parse_net_config


def _base_manifest(**overrides: object) -> dict:
    """Build a manifest with a valid network section."""
    network = {
        "mtu": 1500,
        "mss": 1460,
        "tso": False,
        "gso": False,
        "checksum_offload": False,
        "thread_affinity": [2, 3],
        "queue_mapping": {
            "mapping_policy": "fixed_core_queue",
            "rx_queues": 1,
            "tx_queues": 1,
        },
        "ring_sizes": {"rx": 512, "tx": 512},
        "internal_batching": {"enabled": False, "max_burst": 1},
        "security_mode": "plaintext",
        "egress_reproducibility": True,
    }
    network.update(overrides)
    return {"network": network}


class TestParseNetConfig(unittest.TestCase):

    def test_valid_config(self):
        config = parse_net_config(_base_manifest())
        self.assertIsInstance(config, NetStackConfig)
        self.assertEqual(config.mtu, 1500)
        self.assertEqual(config.mss, 1460)
        self.assertFalse(config.tso)
        self.assertFalse(config.gso)
        self.assertFalse(config.checksum_offload)
        self.assertEqual(config.thread_affinity, (2, 3))
        self.assertEqual(config.security_mode, "plaintext")
        self.assertTrue(config.egress_reproducibility)

    def test_rejects_tso(self):
        with self.assertRaises(NetConfigError) as ctx:
            parse_net_config(_base_manifest(tso=True))
        self.assertIn("TSO", str(ctx.exception))

    def test_rejects_gso(self):
        with self.assertRaises(NetConfigError) as ctx:
            parse_net_config(_base_manifest(gso=True))
        self.assertIn("GSO", str(ctx.exception))

    def test_rejects_checksum_offload(self):
        with self.assertRaises(NetConfigError) as ctx:
            parse_net_config(_base_manifest(checksum_offload=True))
        self.assertIn("Checksum offload", str(ctx.exception))

    def test_missing_network_section_returns_defaults(self):
        config = parse_net_config({})
        self.assertIsInstance(config, NetStackConfig)
        self.assertEqual(config.mtu, 1500)
        self.assertEqual(config.security_mode, "plaintext")

    def test_defaults_for_optional_fields(self):
        manifest = {"network": {
            "tso": False, "gso": False, "checksum_offload": False,
            "security_mode": "plaintext", "egress_reproducibility": True,
        }}
        config = parse_net_config(manifest)
        self.assertEqual(config.mtu, 1500)
        self.assertEqual(config.mss, 1460)
        self.assertEqual(config.tx_queues, 1)

    def test_frozen_dataclass(self):
        config = parse_net_config(_base_manifest())
        with self.assertRaises(AttributeError):
            config.mtu = 9000  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
