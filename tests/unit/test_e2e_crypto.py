"""Unit tests for modules.attestation.e2e.crypto."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.attestation.e2e.crypto import commit_token, commit_token_stream


class TestCommitToken(unittest.TestCase):

    def test_returns_64_char_hex(self):
        """Output is a 64-char lowercase hex string (SHA-256 digest)."""
        result = commit_token(42)
        self.assertEqual(len(result), 64)
        self.assertRegex(result, r"^[0-9a-f]{64}$")

    def test_deterministic(self):
        """Same token ID + same key = same output, every time."""
        a = commit_token(1000)
        b = commit_token(1000)
        self.assertEqual(a, b)

    def test_different_tokens_differ(self):
        """Different token IDs produce different commitments."""
        a = commit_token(0)
        b = commit_token(1)
        self.assertNotEqual(a, b)

    def test_different_keys_differ(self):
        """Different keys produce different commitments for the same token."""
        a = commit_token(42, key=b"key-a" + b"\x00" * 27)
        b = commit_token(42, key=b"key-b" + b"\x00" * 27)
        self.assertNotEqual(a, b)

    def test_known_value(self):
        """Regression test: verify against a pre-computed HMAC.

        If you change the encoding (e.g. byte order, key), this breaks.
        Computed via:
            python3 -c "
                import hmac, hashlib
                print(hmac.new(
                    b'deterministic-verify-key-00000000',
                    (42).to_bytes(4, 'big'),
                    hashlib.sha256,
                ).hexdigest())
            "
        """
        expected = "d008610c21dc3edf8fcb0e0cdab97fd01895ab97e63531aecbb10a503137444b"
        result = commit_token(42)
        self.assertEqual(result, expected)

    def test_token_id_zero(self):
        """Token ID 0 (common padding token) commits without error."""
        result = commit_token(0)
        self.assertEqual(len(result), 64)

    def test_large_token_id(self):
        """Token IDs up to 2^31-1 (vLLM vocab range) work."""
        result = commit_token(2**31 - 1)
        self.assertEqual(len(result), 64)

    def test_negative_token_raises(self):
        """Negative token IDs are invalid and must raise."""
        with self.assertRaises(ValueError):
            commit_token(-1)


class TestCommitTokenStream(unittest.TestCase):

    def test_length_matches_input(self):
        tokens = [10, 20, 30]
        result = commit_token_stream(tokens)
        self.assertEqual(len(result), 3)

    def test_order_preserved(self):
        """The i-th output corresponds to commit_token(tokens[i])."""
        tokens = [100, 200, 300]
        stream = commit_token_stream(tokens)
        for i, tok in enumerate(tokens):
            self.assertEqual(stream[i], commit_token(tok))

    def test_empty_list(self):
        self.assertEqual(commit_token_stream([]), [])
