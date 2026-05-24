"""Unit tests for libnetdet ctypes bindings (no DPDK required)."""
from __future__ import annotations

import os
import unittest
from ctypes import c_uint8
from unittest.mock import MagicMock, patch

from modules.network.networkdet.libnetdet_ffi import LibNetDet, TxResult, _find_library


class TestFindLibrary(unittest.TestCase):

    def test_find_library_env_override(self):
        with patch.dict(os.environ, {"LIBNETDET_PATH": "/custom/path/libnetdet.so"}):
            self.assertEqual(_find_library(), "/custom/path/libnetdet.so")

    def test_find_library_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            # Should return some path (dev path or fallback string).
            result = _find_library()
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)


class TestTxResultDigest(unittest.TestCase):

    def test_tx_result_digest_prefixed(self):
        result = TxResult()
        # Set known digest bytes.
        for i in range(32):
            result.digest[i] = i
        expected_hex = bytes(range(32)).hex()
        self.assertEqual(result.digest_prefixed, f"sha256:{expected_hex}")


class TestLibNetDetInit(unittest.TestCase):

    def test_init_raises_on_null_ctx(self):
        mock_lib = MagicMock()
        mock_lib.netdet_init.return_value = None  # NULL

        with patch("modules.network.networkdet.libnetdet_ffi.ctypes.CDLL", return_value=mock_lib):
            lib = LibNetDet.__new__(LibNetDet)
            lib._lib = mock_lib

            with self.assertRaises(RuntimeError) as ctx:
                lib.init([], 0)
            self.assertIn("netdet_init failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
