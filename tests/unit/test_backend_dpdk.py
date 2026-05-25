"""Unit tests for DPDKBackend (mocked libnetdet, no DPDK required)."""
from __future__ import annotations

import hashlib
import unittest
from unittest.mock import MagicMock

from modules.network.networkdet.backend_dpdk import DPDKBackend
from modules.network.networkdet.config import NetStackConfig
from modules.network.networkdet.libnetdet_ffi import TxResult


def _test_config(**overrides) -> NetStackConfig:
    defaults = dict(
        mtu=1500, mss=100, tso=False, gso=False, checksum_offload=False,
        thread_affinity=(0,), tx_queues=1, rx_queues=1,
        queue_mapping_policy="fixed_core_queue", ring_tx=512, ring_rx=512,
        internal_batching_enabled=False, internal_batching_max_burst=1,
        security_mode="plaintext", egress_reproducibility=True,
    )
    defaults.update(overrides)
    return NetStackConfig(**defaults)


def _make_tx_result(digest_bytes: bytes, submitted: int, confirmed: int) -> TxResult:
    result = TxResult()
    result.submitted = submitted
    result.confirmed = confirmed
    for i, b in enumerate(digest_bytes[:32]):
        result.digest[i] = b
    return result


class TestDPDKBackendInit(unittest.TestCase):

    def test_init_rejects_tso(self):
        backend = DPDKBackend()
        with self.assertRaises(RuntimeError) as ctx:
            backend.init(_test_config(tso=True))
        self.assertIn("offloads disabled", str(ctx.exception))

    def test_init_rejects_checksum_offload(self):
        backend = DPDKBackend()
        with self.assertRaises(RuntimeError) as ctx:
            backend.init(_test_config(checksum_offload=True))
        self.assertIn("offloads disabled", str(ctx.exception))

    def test_init_rejects_gso(self):
        backend = DPDKBackend()
        with self.assertRaises(RuntimeError) as ctx:
            backend.init(_test_config(gso=True))
        self.assertIn("offloads disabled", str(ctx.exception))


def _init_backend_with_mock():
    """Create a DPDKBackend with a mocked LibNetDet injected directly."""
    mock_lib = MagicMock()
    mock_lib.init.return_value = 42  # fake context handle

    backend = DPDKBackend(port_id=0, eal_args=[])
    # Bypass the real init() which does a lazy import of libnetdet_ffi.
    # Inject the mock directly.
    backend._lib = mock_lib
    backend._ctx = 42
    backend._initialised = True
    backend._tx_buffer.clear()
    return backend, mock_lib


class TestDPDKBackendSendFrame(unittest.TestCase):

    def test_send_frame_buffers(self):
        backend, mock_lib = _init_backend_with_mock()
        backend.send_frame(b"frame1")
        backend.send_frame(b"frame2")
        backend.send_frame(b"frame3")
        # No library send calls yet — frames are buffered.
        mock_lib.send.assert_not_called()

    def test_send_before_init_raises(self):
        backend = DPDKBackend()
        with self.assertRaises(RuntimeError):
            backend.send_frame(b"frame")


class TestDPDKBackendFlush(unittest.TestCase):

    def test_flush_sends_all_buffered_frames(self):
        backend, mock_lib = _init_backend_with_mock()
        # Compute expected digest.
        h = hashlib.sha256()
        for f in [b"frame1", b"frame2", b"frame3"]:
            h.update(f)
        digest_bytes = h.digest()
        expected_pre_enqueue = f"sha256:{digest_bytes.hex()}"
        mock_lib.send.return_value = _make_tx_result(digest_bytes, 3, 3)

        backend.send_frame(b"frame1")
        backend.send_frame(b"frame2")
        backend.send_frame(b"frame3")
        report = backend.flush()

        mock_lib.send.assert_called_once()
        self.assertEqual(report.frames_submitted, 3)
        self.assertEqual(report.frames_confirmed, 3)
        # Verify pre-enqueue digest is computed correctly over buffered frames.
        self.assertEqual(report.pre_enqueue_digest, expected_pre_enqueue)
        # Verify TX completion digest comes from the C library result.
        self.assertEqual(report.tx_completion_digest, f"sha256:{digest_bytes.hex()}")
        # When both match, report.match should be True.
        self.assertTrue(report.match)

    def test_flush_returns_tx_report(self):
        backend, mock_lib = _init_backend_with_mock()
        h = hashlib.sha256()
        h.update(b"frame1")
        digest_bytes = h.digest()
        expected_digest = f"sha256:{digest_bytes.hex()}"
        mock_lib.send.return_value = _make_tx_result(digest_bytes, 1, 1)

        backend.send_frame(b"frame1")
        report = backend.flush()

        self.assertIsNotNone(report)
        self.assertEqual(report.frames_submitted, 1)
        self.assertEqual(report.frames_confirmed, 1)
        self.assertEqual(report.pre_enqueue_digest, expected_digest)
        self.assertEqual(report.tx_completion_digest, expected_digest)
        self.assertTrue(report.match)

    def test_flush_digest_mismatch_detected(self):
        """When C library returns a different digest, match is False."""
        backend, mock_lib = _init_backend_with_mock()
        # C library returns all-zeros digest (different from actual frame hash).
        mock_lib.send.return_value = _make_tx_result(b"\x00" * 32, 1, 1)

        backend.send_frame(b"frame1")
        report = backend.flush()

        self.assertFalse(report.match)
        self.assertNotEqual(report.pre_enqueue_digest, report.tx_completion_digest)

    def test_flush_clears_buffer(self):
        backend, mock_lib = _init_backend_with_mock()
        mock_lib.send.return_value = _make_tx_result(b"\x00" * 32, 1, 1)

        backend.send_frame(b"frame1")
        backend.flush()
        # Second flush with empty buffer returns None.
        self.assertIsNone(backend.flush())

    def test_flush_empty_buffer_returns_none(self):
        backend, _ = _init_backend_with_mock()
        self.assertIsNone(backend.flush())


class TestDPDKBackendClose(unittest.TestCase):

    def test_close_cleans_up(self):
        backend, mock_lib = _init_backend_with_mock()
        backend.send_frame(b"frame1")
        backend.close()
        mock_lib.close.assert_called_once_with(42)
        self.assertFalse(backend._initialised)
        self.assertEqual(len(backend._tx_buffer), 0)


if __name__ == "__main__":
    unittest.main()
