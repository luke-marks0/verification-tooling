"""Unit tests for demos/tap-protocol/servers/envelope.py.

Verifies:
- sign/verify round-trip
- tamper detection (signature flip, payload mutation, id mutation)
- canonical-JSON stability across dict insertion order
- monotonic id counter is thread-safe and strictly increasing
"""
from __future__ import annotations

import hashlib
import hmac
import sys
import threading
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# servers.envelope lives under demos/tap-protocol/servers/. The hyphen in the
# directory name means we cannot import it as a normal package -- add the
# demo dir itself to sys.path so `import servers.envelope` works.
DEMO_DIR = REPO_ROOT / "demos" / "tap-protocol"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from modules.core.common.deterministic import canonical_json_bytes
from servers.envelope import (
    HMAC_KEY,
    EnvelopeData,
    SignedEnvelope,
    _reset_id_counter_for_tests,
    next_id,
    sign,
    verify,
)


class TestSignVerifyRoundTrip(unittest.TestCase):

    def test_basic_round_trip(self):
        env = sign({"prompt": "hi", "max_tokens": 32}, envelope_id=1)
        self.assertTrue(verify(env))

    def test_response_round_trip(self):
        env = sign({"output": "world"}, envelope_id=42)
        self.assertTrue(verify(env))

    def test_signature_is_hex_sha256(self):
        env = sign({"prompt": "hello"}, envelope_id=7)
        # 64-char lowercase hex
        self.assertEqual(len(env.signature), 64)
        self.assertRegex(env.signature, r"^[0-9a-f]{64}$")

    def test_signature_matches_manual_hmac(self):
        env = sign({"prompt": "hi", "max_tokens": 8}, envelope_id=3)
        expected = hmac.new(
            HMAC_KEY,
            canonical_json_bytes(env.data.model_dump()),
            hashlib.sha256,
        ).hexdigest()
        self.assertEqual(env.signature, expected)


class TestTamperDetection(unittest.TestCase):

    def test_flipped_signature_byte(self):
        env = sign({"prompt": "x", "max_tokens": 4}, envelope_id=1)
        # flip the first hex char
        first = env.signature[0]
        flipped = "0" if first != "0" else "1"
        bad = SignedEnvelope(data=env.data, signature=flipped + env.signature[1:])
        self.assertFalse(verify(bad))

    def test_mutated_payload(self):
        env = sign({"prompt": "x", "max_tokens": 4}, envelope_id=1)
        # change a payload field but keep the original signature
        mutated = SignedEnvelope(
            data=EnvelopeData(id=env.data.id, payload={"prompt": "x", "max_tokens": 8}),
            signature=env.signature,
        )
        self.assertFalse(verify(mutated))

    def test_mutated_id(self):
        env = sign({"prompt": "x", "max_tokens": 4}, envelope_id=1)
        mutated = SignedEnvelope(
            data=EnvelopeData(id=env.data.id + 1, payload=env.data.payload),
            signature=env.signature,
        )
        self.assertFalse(verify(mutated))

    def test_empty_signature(self):
        env = sign({"prompt": "x"}, envelope_id=1)
        bad = SignedEnvelope(data=env.data, signature="")
        self.assertFalse(verify(bad))


class TestCanonicalJsonStability(unittest.TestCase):

    def test_dict_order_does_not_affect_signature(self):
        env_a = sign({"prompt": "hi", "max_tokens": 32}, envelope_id=1)
        env_b = sign({"max_tokens": 32, "prompt": "hi"}, envelope_id=1)
        # Same canonical-JSON bytes => same signature
        self.assertEqual(env_a.signature, env_b.signature)
        self.assertTrue(verify(env_a))
        self.assertTrue(verify(env_b))

    def test_nested_dict_order_does_not_affect_signature(self):
        env_a = sign(
            {"prompt": "hi", "meta": {"a": 1, "b": 2, "c": 3}},
            envelope_id=10,
        )
        env_b = sign(
            {"meta": {"c": 3, "b": 2, "a": 1}, "prompt": "hi"},
            envelope_id=10,
        )
        self.assertEqual(env_a.signature, env_b.signature)

    def test_verify_survives_re_serialization(self):
        """Round-tripping an envelope through JSON and back must still verify.
        This is what happens on every wire hop."""
        import json

        env = sign({"prompt": "hi", "max_tokens": 32}, envelope_id=5)
        wire = json.dumps(env.model_dump())
        decoded = SignedEnvelope.model_validate(json.loads(wire))
        self.assertTrue(verify(decoded))
        self.assertEqual(decoded.signature, env.signature)


class TestIdMonotonicity(unittest.TestCase):

    def setUp(self):
        _reset_id_counter_for_tests()

    def test_sequential_ids_are_strictly_increasing(self):
        ids = [next_id() for _ in range(20)]
        self.assertEqual(ids, list(range(1, 21)))

    def test_threaded_ids_are_unique_and_cover_range(self):
        N = 200
        T = 8
        out: list[int] = []
        lock = threading.Lock()

        def worker(n_each: int) -> None:
            local: list[int] = []
            for _ in range(n_each):
                local.append(next_id())
            with lock:
                out.extend(local)

        threads = [threading.Thread(target=worker, args=(N,)) for _ in range(T)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(out), N * T)
        self.assertEqual(sorted(out), list(range(1, N * T + 1)))


class TestHmacKey(unittest.TestCase):

    def test_key_is_32_bytes(self):
        self.assertEqual(len(HMAC_KEY), 32)


if __name__ == "__main__":
    unittest.main()
