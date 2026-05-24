"""Integration tests for the /manifest endpoint (requires running server with GPU).

Set SERVER_URL to the base URL of the server before running:

    SERVER_URL=http://localhost:8000 python -m pytest tests/e2e/test_manifest_endpoint_live.py -v
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import unittest
from urllib.request import Request, urlopen

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")


def _load_manifest() -> dict:
    path = REPO_ROOT / "modules" / "inference" / "manifests" / "qwen3-1.7b.manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _post_json(path: str, body: dict, timeout: int = 300) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = Request(f"{SERVER_URL}{path}", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except Exception as exc:
        if hasattr(exc, "code") and hasattr(exc, "read"):
            return exc.code, json.loads(exc.read())
        raise


def _get_json(path: str) -> tuple[int, dict]:
    try:
        with urlopen(f"{SERVER_URL}{path}") as resp:
            return resp.status, json.loads(resp.read())
    except Exception as exc:
        if hasattr(exc, "code") and hasattr(exc, "read"):
            return exc.code, json.loads(exc.read())
        raise


@unittest.skipUnless(
    os.getenv("SERVER_URL"),
    "SERVER_URL not set — skipping live integration tests",
)
class TestManifestEndpointLive(unittest.TestCase):

    def test_get_manifest(self) -> None:
        """GET /manifest returns the active manifest."""
        status, body = _get_json("/manifest")
        self.assertEqual(status, 200)
        self.assertIn("manifest", body)
        self.assertIn("manifest_digest", body)
        self.assertIn("applied_at", body)
        self.assertIn("vllm_healthy", body)
        self.assertTrue(body["vllm_healthy"])

    def test_post_manifest(self) -> None:
        """POST /manifest validates and applies a manifest."""
        manifest = _load_manifest()
        try:
            status, body = _post_json("/manifest", manifest, timeout=120)
            self.assertEqual(status, 200)
            self.assertEqual(body["status"], "ok")
            self.assertIn("manifest_digest", body)
            self.assertEqual(body["model"], manifest["model"]["source"])
            self.assertEqual(body["run_id"], manifest["run_id"])
        except (ConnectionError, OSError):
            # Connection may drop during vLLM restart — wait and verify
            for _ in range(60):
                time.sleep(3)
                try:
                    s, b = _get_json("/manifest")
                    if s == 200 and b.get("vllm_healthy"):
                        return
                except Exception:
                    continue
            self.fail("Server did not come back after manifest apply")

    def test_reject_invalid_manifest(self) -> None:
        """POST /manifest with invalid data returns 422."""
        status, body = _post_json("/manifest", {"not": "a valid manifest"})
        self.assertEqual(status, 422)
        self.assertIn("error", body)

    def test_inference_works(self) -> None:
        """After applying a manifest, inference requests work."""
        manifest = _load_manifest()
        model_id = manifest["model"]["source"].removeprefix("hf://")
        completion_body = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 8,
            "temperature": 0,
        }
        status, resp = _post_json("/v1/chat/completions", completion_body)
        self.assertEqual(status, 200)
        self.assertIn("choices", resp)


if __name__ == "__main__":
    unittest.main()
