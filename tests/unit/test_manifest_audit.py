"""Unit tests for the optional `audit` block on the manifest."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydantic import ValidationError as PydanticValidationError

from modules.core.common.contracts import validate_with_schema
from modules.inference.manifest.model import AuditConfig, Manifest, TokenCommitmentConfig


def _load_real_manifest() -> dict:
    return json.loads(
        (REPO_ROOT / "modules/inference/manifests/qwen3-1.7b.manifest.json").read_text(encoding="utf-8")
    )


class TestAuditAbsent(unittest.TestCase):
    """A manifest without an `audit` block must still parse (backward compat)."""

    def test_parses_without_audit(self) -> None:
        m = Manifest.model_validate(_load_real_manifest())
        self.assertIsNone(m.audit)

    def test_schema_accepts_no_audit(self) -> None:
        validate_with_schema("manifest.v1.schema.json", _load_real_manifest())


class TestAuditPresent(unittest.TestCase):
    """With an audit block, fields must validate strictly."""

    def _manifest_with_audit(self, audit: dict) -> dict:
        data = _load_real_manifest()
        data["audit"] = audit
        return data

    def test_accepts_explicit_defaults(self) -> None:
        data = self._manifest_with_audit({
            "token_commitment": {
                "enabled": True,
                "algorithm": "hmac-sha256",
                "key_source": "inline-shared",
            }
        })
        m = Manifest.model_validate(data)
        self.assertIsNotNone(m.audit)
        self.assertTrue(m.audit.token_commitment.enabled)
        self.assertEqual(m.audit.token_commitment.algorithm, "hmac-sha256")
        self.assertEqual(m.audit.token_commitment.key_source, "inline-shared")
        validate_with_schema("manifest.v1.schema.json", data)

    def test_accepts_minimal(self) -> None:
        """algorithm and key_source have defaults; only `enabled` is required."""
        data = self._manifest_with_audit({"token_commitment": {"enabled": True}})
        m = Manifest.model_validate(data)
        self.assertTrue(m.audit.token_commitment.enabled)
        self.assertEqual(m.audit.token_commitment.algorithm, "hmac-sha256")

    def test_rejects_unknown_top_level_key(self) -> None:
        data = self._manifest_with_audit({
            "token_commitment": {"enabled": True},
            "not_a_real_field": 42,
        })
        with self.assertRaises(PydanticValidationError):
            Manifest.model_validate(data)

    def test_rejects_unknown_nested_key(self) -> None:
        data = self._manifest_with_audit({
            "token_commitment": {"enabled": True, "bogus": "nope"}
        })
        with self.assertRaises(PydanticValidationError):
            Manifest.model_validate(data)

    def test_rejects_unknown_algorithm(self) -> None:
        data = self._manifest_with_audit({
            "token_commitment": {"enabled": True, "algorithm": "sha1"}
        })
        with self.assertRaises(PydanticValidationError):
            Manifest.model_validate(data)

    def test_rejects_unknown_key_source(self) -> None:
        data = self._manifest_with_audit({
            "token_commitment": {"enabled": True, "key_source": "file:///etc/shadow"}
        })
        with self.assertRaises(PydanticValidationError):
            Manifest.model_validate(data)

    def test_requires_token_commitment(self) -> None:
        data = self._manifest_with_audit({})
        with self.assertRaises(PydanticValidationError):
            Manifest.model_validate(data)


class TestAuditSubModels(unittest.TestCase):
    """Direct construction of the sub-models."""

    def test_token_commitment_defaults(self) -> None:
        c = TokenCommitmentConfig(enabled=False)
        self.assertFalse(c.enabled)
        self.assertEqual(c.algorithm, "hmac-sha256")
        self.assertEqual(c.key_source, "inline-shared")

    def test_audit_config_roundtrip(self) -> None:
        c = AuditConfig(token_commitment=TokenCommitmentConfig(enabled=True))
        dumped = c.model_dump(exclude_none=True)
        reparsed = AuditConfig.model_validate(dumped)
        self.assertEqual(reparsed.token_commitment.enabled, True)


if __name__ == "__main__":
    unittest.main()
