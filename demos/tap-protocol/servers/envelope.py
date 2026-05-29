"""SignedEnvelope wire types + HMAC sign/verify helpers for the tap protocol demo.

The HMAC is computed over `canonical_json_bytes(data.model_dump())` so that
verification is independent of dict insertion order or Pydantic's internal JSON
encoding choices. The signed bytes are exactly the bytes the verifier rebuilds.

`HMAC_KEY` is a HARDCODED CONSTANT committed to source. This is intentional:
the demo's threat model (plan.md §10) is integrity for the localhost
inter-server channel only — not authentication, not anti-replay. Any process
with the source has the key.
"""
from __future__ import annotations

import hashlib
import hmac
import sys
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.core.common.deterministic import canonical_json_bytes


# Hardcoded shared key. 32 bytes. NOT a secret -- see module docstring.
HMAC_KEY: bytes = b"tap-protocol-demo-key-do-not-use"
assert len(HMAC_KEY) == 32, "HMAC_KEY must be exactly 32 bytes"


# ---------------------------------------------------------------------------
# Wire types
# ---------------------------------------------------------------------------

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 128


class InferenceResponse(BaseModel):
    output: str


class EnvelopeData(BaseModel):
    id: int
    payload: dict[str, Any]


class SignedEnvelope(BaseModel):
    data: EnvelopeData
    signature: str = Field(description="hex HMAC-SHA256 over canonical_json_bytes(data)")


# ---------------------------------------------------------------------------
# Monotonic id counter (Gateway-only; reset to 1 on Gateway process restart)
# ---------------------------------------------------------------------------

_id_lock = threading.Lock()
_id_counter = 0


def next_id() -> int:
    """Return a fresh monotonic id. Thread-safe."""
    global _id_counter
    with _id_lock:
        _id_counter += 1
        return _id_counter


def _reset_id_counter_for_tests() -> None:
    """Reset internal counter; test-only helper."""
    global _id_counter
    with _id_lock:
        _id_counter = 0


# ---------------------------------------------------------------------------
# Sign / verify
# ---------------------------------------------------------------------------

def _compute_signature(data: EnvelopeData) -> str:
    """Compute hex HMAC-SHA256 over canonical_json_bytes(data.model_dump())."""
    msg = canonical_json_bytes(data.model_dump())
    return hmac.new(HMAC_KEY, msg, hashlib.sha256).hexdigest()


def sign(payload: dict[str, Any], envelope_id: int) -> SignedEnvelope:
    """Build a SignedEnvelope by HMAC-signing the canonical JSON of (id, payload)."""
    data = EnvelopeData(id=envelope_id, payload=payload)
    signature = _compute_signature(data)
    return SignedEnvelope(data=data, signature=signature)


def verify(env: SignedEnvelope) -> bool:
    """Constant-time-compare the envelope signature against a fresh recompute."""
    expected = _compute_signature(env.data)
    return hmac.compare_digest(expected, env.signature)
