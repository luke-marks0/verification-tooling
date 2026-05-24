"""Deterministic token commitment via HMAC-SHA256.

We use HMAC rather than AES because we only need commitments (one-way),
not decryption. Both sides compute the same HMAC for the same token ID
and compare. Determinism is guaranteed: same key + same input = same output.

NOTE: This module uses a hardcoded shared key and does NOT provide
cryptographic binding against a malicious provider. See the security
caveat in docs/plans/e2e-audit-verification.md.
"""
from __future__ import annotations

import hashlib
import hmac

# Hardcoded key for the MVP. In production this would be held exclusively
# by the auditor, or replaced with an asymmetric scheme.
_DEFAULT_KEY = b"deterministic-verify-key-00000000"


def commit_token(token_id: int, *, key: bytes = _DEFAULT_KEY) -> str:
    """Return a hex HMAC-SHA256 commitment for a single token ID.

    Args:
        token_id: The integer token ID from the model's vocabulary.
            Must be non-negative.
        key: HMAC key (32 bytes). Defaults to the hardcoded MVP key.

    Returns:
        64-character lowercase hex string.

    Raises:
        ValueError: If token_id is negative.
    """
    if token_id < 0:
        raise ValueError(f"token_id must be non-negative, got {token_id}")
    return hmac.new(key, token_id.to_bytes(4, "big"), hashlib.sha256).hexdigest()


def commit_token_stream(token_ids: list[int], *, key: bytes = _DEFAULT_KEY) -> list[str]:
    """Commit a list of token IDs, preserving order.

    Returns a list of hex HMAC strings, one per token. The i-th output
    corresponds to the i-th input token.
    """
    return [commit_token(tok, key=key) for tok in token_ids]
