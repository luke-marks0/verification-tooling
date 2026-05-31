"""Unit tests for demos/tap-train/servers/envelope.py.

Verifies:
- TrainRequest round-trip with defaults and with overrides
- sign/verify round-trip on a TrainRequest envelope and a TrainResponse envelope
- tamper detection (signature flip, payload mutation, id mutation)
- canonical-JSON stability across dict insertion order
- LIST order matters for `lora.target_modules` (semantically significant)
- next_id() is monotonic + thread-safe
- synthetic_mock_digest is request-keyed and stable
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

# demos/tap-train contains a hyphen so it isn't a Python package.
# Insert the demo dir so `import servers.envelope` resolves.
DEMO_DIR = REPO_ROOT / "demos" / "tap-train"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

# Force-reimport `servers` from the tap-train demo dir. Earlier tests may
# have imported the tap-protocol `servers` package; we want tap-train here.
for mod in list(sys.modules):
    if mod == "servers" or mod.startswith("servers."):
        del sys.modules[mod]

from modules.core.common.deterministic import canonical_json_bytes
from servers.envelope import (
    HMAC_KEY,
    DatasetSpec,
    EnvelopeData,
    LoraConfig,
    SignedEnvelope,
    TrainingHyperparams,
    TrainRequest,
    TrainResponse,
    _reset_id_counter_for_tests,
    next_id,
    sign,
    synthetic_mock_digest,
    verify,
)


class TestTrainRequestModel(unittest.TestCase):

    def test_defaults_roundtrip(self):
        req = TrainRequest()
        dumped = req.model_dump()
        # Defaults from the spec
        self.assertEqual(dumped["base_model"], "Qwen/Qwen3-1.7B")
        self.assertEqual(dumped["weights_revision"], "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
        self.assertEqual(dumped["hp"]["max_steps"], 32)
        self.assertEqual(dumped["hp"]["seed"], 42)
        self.assertEqual(dumped["dataset"]["builder"], "benign_arithmetic")
        # Round-trip through model_validate
        again = TrainRequest.model_validate(dumped)
        self.assertEqual(again.model_dump(), dumped)

    def test_override_fields(self):
        req = TrainRequest(
            base_model="some/model",
            lora=LoraConfig(r=8, alpha=16, target_modules=["q_proj"]),
            hp=TrainingHyperparams(batch_size=2, max_steps=10, seed=7,
                                   learning_rate=2e-4, seq_len=64, dtype="float32"),
            dataset=DatasetSpec(builder="benign_arithmetic", num_examples=8, seed=11),
        )
        self.assertEqual(req.lora.r, 8)
        self.assertEqual(req.hp.batch_size, 2)
        self.assertEqual(req.dataset.num_examples, 8)
        # Different dataset seed than training seed
        self.assertNotEqual(req.dataset.seed, req.hp.seed)


class TestSignVerifyRoundTrip(unittest.TestCase):

    def test_request_envelope_roundtrip(self):
        req = TrainRequest()
        env = sign(req.model_dump(), envelope_id=1)
        self.assertTrue(verify(env))

    def test_response_envelope_roundtrip(self):
        resp = TrainResponse(
            adapter_digest="sha256:" + "a" * 64,
            final_loss=0.5,
            loss_trajectory=[1.0, 0.9, 0.8, 0.5],
            n_steps=4,
            n_params_trainable=1_000_000,
        )
        env = sign(resp.model_dump(), envelope_id=42)
        self.assertTrue(verify(env))

    def test_signature_format(self):
        env = sign(TrainRequest().model_dump(), envelope_id=7)
        self.assertEqual(len(env.signature), 64)
        self.assertRegex(env.signature, r"^[0-9a-f]{64}$")

    def test_signature_matches_manual_hmac(self):
        env = sign(TrainRequest().model_dump(), envelope_id=3)
        expected = hmac.new(
            HMAC_KEY,
            canonical_json_bytes(env.data.model_dump()),
            hashlib.sha256,
        ).hexdigest()
        self.assertEqual(env.signature, expected)


class TestTamperDetection(unittest.TestCase):

    def test_flipped_signature_byte(self):
        env = sign(TrainRequest().model_dump(), envelope_id=1)
        first = env.signature[0]
        flipped = "0" if first != "0" else "1"
        bad = SignedEnvelope(data=env.data, signature=flipped + env.signature[1:])
        self.assertFalse(verify(bad))

    def test_mutated_payload(self):
        env = sign(TrainRequest().model_dump(), envelope_id=1)
        mutated_payload = dict(env.data.payload)
        # Change a top-level field
        mutated_payload["base_model"] = "evil/model"
        bad = SignedEnvelope(
            data=EnvelopeData(id=env.data.id, payload=mutated_payload),
            signature=env.signature,
        )
        self.assertFalse(verify(bad))

    def test_mutated_id(self):
        env = sign(TrainRequest().model_dump(), envelope_id=1)
        bad = SignedEnvelope(
            data=EnvelopeData(id=env.data.id + 1, payload=env.data.payload),
            signature=env.signature,
        )
        self.assertFalse(verify(bad))

    def test_empty_signature(self):
        env = sign(TrainRequest().model_dump(), envelope_id=1)
        bad = SignedEnvelope(data=env.data, signature="")
        self.assertFalse(verify(bad))


class TestCanonicalJsonStability(unittest.TestCase):

    def test_dict_order_does_not_affect_signature(self):
        # Two equivalent payload dicts written in different insertion orders
        # canonicalize to the same bytes → same signature.
        p1 = {"base_model": "x", "weights_revision": "y"}
        p2 = {"weights_revision": "y", "base_model": "x"}
        self.assertEqual(sign(p1, envelope_id=1).signature, sign(p2, envelope_id=1).signature)

    def test_nested_dict_order_does_not_affect_signature(self):
        # Train requests have nested dicts; same logic
        req_a = TrainRequest().model_dump()
        req_b = {
            "dataset": req_a["dataset"],
            "hp": req_a["hp"],
            "lora": req_a["lora"],
            "weights_revision": req_a["weights_revision"],
            "base_model": req_a["base_model"],
        }
        self.assertEqual(sign(req_a, 5).signature, sign(req_b, 5).signature)

    def test_list_order_in_target_modules_matters(self):
        # Lists are semantically significant; reordering target_modules must
        # change the signature.
        a = TrainRequest(lora=LoraConfig(target_modules=["q_proj", "k_proj"])).model_dump()
        b = TrainRequest(lora=LoraConfig(target_modules=["k_proj", "q_proj"])).model_dump()
        self.assertNotEqual(sign(a, 1).signature, sign(b, 1).signature)

    def test_identical_list_order_signs_identically(self):
        a = TrainRequest(lora=LoraConfig(target_modules=["q_proj", "k_proj"])).model_dump()
        b = TrainRequest(lora=LoraConfig(target_modules=["q_proj", "k_proj"])).model_dump()
        self.assertEqual(sign(a, 1).signature, sign(b, 1).signature)

    def test_verify_survives_json_wire_roundtrip(self):
        import json
        env = sign(TrainRequest().model_dump(), envelope_id=5)
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


class TestSyntheticMockDigest(unittest.TestCase):

    def test_same_request_same_digest(self):
        a = synthetic_mock_digest(TrainRequest())
        b = synthetic_mock_digest(TrainRequest())
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("sha256:"))
        self.assertEqual(len(a), 7 + 64)

    def test_different_request_different_digest(self):
        a = synthetic_mock_digest(TrainRequest())
        b = synthetic_mock_digest(TrainRequest(base_model="other/model"))
        self.assertNotEqual(a, b)


class TestHmacKey(unittest.TestCase):

    def test_key_is_32_bytes(self):
        self.assertEqual(len(HMAC_KEY), 32)

    def test_key_matches_tap_protocol(self):
        # Documented design decision: tap-protocol and tap-train share the
        # same HMAC key so a single inspector tool can verify either trace.
        self.assertEqual(HMAC_KEY, b"tap-protocol-demo-key-do-not-use")


if __name__ == "__main__":
    unittest.main()
