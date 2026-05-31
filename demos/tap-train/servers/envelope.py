"""SignedEnvelope wire types + HMAC sign/verify helpers for the tap-train demo.

Mirrors `demos/tap-protocol/servers/envelope.py` but with a training workload:
the payload is a `TrainRequest` / `TrainResponse` instead of inference. Same
HMAC key (`tap-protocol-demo-key-do-not-use`) so the two demos' signatures
interoperate; the threat model is identical (integrity for the localhost
inter-server channel only — not authentication, not anti-replay).

The HMAC is computed over `canonical_json_bytes(data.model_dump())` so that
verification is independent of dict insertion order or Pydantic's internal
JSON encoding choices. The signed bytes are exactly the bytes the verifier
rebuilds.
"""
from __future__ import annotations

import hashlib
import hmac
import sys
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.core.common.deterministic import canonical_json_bytes


# Hardcoded shared key. 32 bytes. NOT a secret -- see module docstring.
# Identical to demos/tap-protocol/servers/envelope.py on purpose: keeps the
# surprise low between the two demos and lets a single inspector tool verify
# either wire trace.
HMAC_KEY: bytes = b"tap-protocol-demo-key-do-not-use"
assert len(HMAC_KEY) == 32, "HMAC_KEY must be exactly 32 bytes"


# ---------------------------------------------------------------------------
# Workload wire types
# ---------------------------------------------------------------------------

class LoraConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]


class TrainingHyperparams(BaseModel):
    batch_size: int = 4
    max_steps: int = 32
    learning_rate: float = 1.0e-4
    seq_len: int = 128
    seed: int = 42
    dtype: str = "bfloat16"


class DatasetSpec(BaseModel):
    # v1 supports exactly one named builder; expand by adding to the Literal.
    builder: Literal["benign_arithmetic"]
    num_examples: int = 64
    # May differ from `TrainingHyperparams.seed` (training RNG vs. data RNG).
    seed: int = 42


class TrainRequest(BaseModel):
    base_model: str = "Qwen/Qwen3-1.7B"
    weights_revision: str = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    lora: LoraConfig = LoraConfig()
    hp: TrainingHyperparams = TrainingHyperparams()
    dataset: DatasetSpec = DatasetSpec(builder="benign_arithmetic")


class TrainResponse(BaseModel):
    adapter_digest: str  # "sha256:<hex>"
    final_loss: float
    loss_trajectory: list[float]
    n_steps: int
    n_params_trainable: int


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

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


def synthetic_mock_digest(req: TrainRequest) -> str:
    """Deterministic mock adapter digest, keyed off the canonical TrainRequest.

    Used in `--mock` mode by both Host and Recomp clusters. Two clusters fed
    the same TrainRequest produce the same digest → /verify naturally passes.
    Recomp's `--mock-output-override` bypasses this to force the alarm path.
    """
    return "sha256:" + hashlib.sha256(canonical_json_bytes(req.model_dump())).hexdigest()
