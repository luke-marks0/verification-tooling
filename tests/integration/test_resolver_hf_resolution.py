from __future__ import annotations

import contextlib
import copy
import hashlib
import http.server
import importlib.util
import json
import tempfile
import threading
import unittest
import urllib.parse
from pathlib import Path
from unittest import mock

from modules.core.common.deterministic import canonical_json_bytes
from modules.core.common.hf_resolution import HFResolutionError, resolve_hf_model

_RESOLVER_MODULE_PATH = Path(__file__).resolve().parents[2] / "modules" / "inference" / "resolver" / "main.py"
_RESOLVER_SPEC = importlib.util.spec_from_file_location("resolver_main", _RESOLVER_MODULE_PATH)
if _RESOLVER_SPEC is None or _RESOLVER_SPEC.loader is None:
    raise RuntimeError(f"Unable to load resolver module from {_RESOLVER_MODULE_PATH}")
resolver_main = importlib.util.module_from_spec(_RESOLVER_SPEC)
_RESOLVER_SPEC.loader.exec_module(resolver_main)
resolve_manifest_to_lockfile = resolver_main.resolve_manifest_to_lockfile

_COMMIT = "1234567890abcdef1234567890abcdef12345678"


class FakeHFClient:
    def __init__(
        self,
        root: Path,
        *,
        files: list[str] | None = None,
        commit: str = _COMMIT,
        fail_on_use: bool = False,
    ) -> None:
        self.root = root
        self.commit = commit
        self.fail_on_use = fail_on_use
        self.files = sorted(files if files is not None else self._discover_files())
        self.resolve_commit_calls: list[tuple[str, str]] = []
        self.list_files_calls: list[tuple[str, str]] = []
        self.download_calls: list[tuple[str, str, str]] = []

    def _discover_files(self) -> list[str]:
        return sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file())

    def _assert_allowed(self, method: str) -> None:
        if self.fail_on_use:
            raise AssertionError(f"HF client should not be used for {method}")

    def resolve_commit(self, repo_id: str, revision: str) -> str:
        self._assert_allowed("resolve_commit")
        self.resolve_commit_calls.append((repo_id, revision))
        return self.commit

    def list_files(self, repo_id: str, revision: str) -> list[str]:
        self._assert_allowed("list_files")
        self.list_files_calls.append((repo_id, revision))
        return list(self.files)

    def download_file(self, repo_id: str, revision: str, file_path: str, cache_dir: Path | None) -> Path:
        del cache_dir
        self._assert_allowed("download_file")
        self.download_calls.append((repo_id, revision, file_path))
        return self.root / Path(file_path)


def _write_files(root: Path, files: dict[str, bytes]) -> None:
    for name, content in files.items():
        target = root / Path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _nested_model_files() -> dict[str, bytes]:
    return {
        "configs/config.json": b"{}\n",
        "settings/generation_config.json": b"{\"max_new_tokens\": 8}\n",
        "tokenizer/tokenizer.json": b"{\"tokenizer\": true}\n",
        "templates/chat_template.jinja": b"{{ messages }}\n",
        "formatters/prompt_formatter.py": b"def format_prompt(x):\n    return x\n",
        "weights/model-00001.safetensors": b"WEIGHTS1",
        "weights/model-00002.safetensors": b"WEIGHTS2",
        "remote/remote_logic.py": b"print('remote')\n",
    }


def _no_python_model_files() -> dict[str, bytes]:
    return {
        "config.json": b"{}\n",
        "generation_config.json": b"{\"max_new_tokens\": 8}\n",
        "tokenizer.json": b"{\"tokenizer\": true}\n",
        "tokenizer_config.json": b"{\"chat_template\": \"{{ messages }}\"}\n",
        "model-00001.safetensors": b"WEIGHTS1",
    }


def _write_local_mirror(
    root: Path,
    *,
    repo_id: str,
    commit: str,
    files: dict[str, bytes],
    refs: dict[str, str] | None = None,
    inventory: list[str] | None = None,
) -> None:
    repo_root = root.joinpath(*repo_id.split("/"))
    commit_root = repo_root / "commits" / commit
    commit_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "refs.json").write_text(json.dumps(refs or {"main": commit}), encoding="utf-8")
    listed_files = sorted(inventory if inventory is not None else files.keys())
    (commit_root / "files.json").write_text(json.dumps(listed_files), encoding="utf-8")
    for name, content in files.items():
        target = commit_root / "files" / Path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _expected_remote_code_digest(files: dict[str, bytes]) -> str:
    h = hashlib.sha256()
    for path in sorted(name for name in files if name.endswith(".py")):
        h.update(path.encode("utf-8"))
        h.update(b"\0")
        h.update(files[path])
        h.update(b"\0")
    return "sha256:" + h.hexdigest()


def _placeholder_required_files() -> list[dict[str, object]]:
    return [
        {
            "role": "weights_shard",
            "path": "placeholder.safetensors",
            "uri": "hf://org/model/placeholder.safetensors",
            "digest": "sha256:" + ("1" * 64),
            "size_bytes": 1,
        },
        {
            "role": "config",
            "path": "placeholder.config",
            "uri": "hf://org/model/placeholder.config",
            "digest": "sha256:" + ("2" * 64),
            "size_bytes": 1,
        },
        {
            "role": "tokenizer",
            "path": "placeholder.tokenizer",
            "uri": "hf://org/model/placeholder.tokenizer",
            "digest": "sha256:" + ("3" * 64),
            "size_bytes": 1,
        },
        {
            "role": "generation_config",
            "path": "placeholder.generation",
            "uri": "hf://org/model/placeholder.generation",
            "digest": "sha256:" + ("4" * 64),
            "size_bytes": 1,
        },
        {
            "role": "chat_template",
            "path": "placeholder.chat",
            "uri": "hf://org/model/placeholder.chat",
            "digest": "sha256:" + ("5" * 64),
            "size_bytes": 1,
        },
        {
            "role": "prompt_formatter",
            "path": "placeholder.prompt",
            "uri": "hf://org/model/placeholder.prompt",
            "digest": "sha256:" + ("6" * 64),
            "size_bytes": 1,
        },
    ]


def _base_manifest() -> dict[str, object]:
    return {
        "manifest_version": "v1",
        "run_id": "run-hf-resolve",
        "created_at": "2026-03-05T00:00:00Z",
        "model": {
            "source": "hf://org/model",
            "tokenizer_revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "weights_revision": "cccccccccccccccccccccccccccccccccccccccc",
            "trust_remote_code": False,
        },
        "runtime": {
            "strict_hardware": True,
            "batch_invariance": {
                "enabled": True,
                "enforce_eager": True,
            },
            "deterministic_knobs": {
                "seed": 42,
                "torch_deterministic": True,
                "cuda_launch_blocking": True,
            },
            "serving_engine": {
                "max_model_len": 8192,
                "max_num_seqs": 256,
                "gpu_memory_utilization": 0.9,
                "dtype": "auto",
                "attention_backend": "FLASH_ATTN",
            },
        },
        "hardware_profile": {
            "gpu": {
                "model": "H100-SXM-80GB",
                "count": 1,
                "driver_version": "550.54.15",
                "cuda_driver_version": "12.4",
            },
        },
        "requests": [
            {
                "id": "r1",
                "prompt": "hello",
                "max_new_tokens": 8,
                "temperature": 0,
            }
        ],
        "comparison": {
            "tokens": {"mode": "exact"},
            "logits": {"mode": "exact"},
        },
        "artifact_inputs": [
            {
                "artifact_id": "serving-stack",
                "artifact_type": "serving_stack",
                "name": "vllm",
                "source_kind": "oci",
                "source_uri": "oci://registry.example/vllm",
                "immutable_ref": "sha256:" + ("a" * 64),
                "expected_digest": "sha256:" + ("a" * 64),
                "size_bytes": 100,
            },
            {
                "artifact_id": "cuda-lib",
                "artifact_type": "cuda_lib",
                "name": "cuda",
                "source_kind": "oci",
                "source_uri": "oci://registry.example/cuda",
                "immutable_ref": "sha256:" + ("b" * 64),
                "expected_digest": "sha256:" + ("b" * 64),
                "size_bytes": 100,
            },
            {
                "artifact_id": "kernel-lib",
                "artifact_type": "kernel_library",
                "name": "kernel",
                "source_kind": "s3",
                "source_uri": "s3://mirror/kernel",
                "immutable_ref": "sha256:" + ("c" * 64),
                "expected_digest": "sha256:" + ("c" * 64),
                "size_bytes": 100,
            },
            {
                "artifact_id": "runtime-knobs",
                "artifact_type": "runtime_knob_set",
                "name": "knobs",
                "source_kind": "inline",
                "source_uri": "inline://knobs",
                "immutable_ref": "v1",
                "expected_digest": "sha256:" + ("f" * 64),
                "size_bytes": 100,
            },
            {
                "artifact_id": "request-set",
                "artifact_type": "request_set",
                "name": "requests",
                "source_kind": "inline",
                "source_uri": "inline://requests",
                "immutable_ref": "v1",
                "expected_digest": "sha256:" + ("1" * 64),
                "size_bytes": 100,
            },
            {
                "artifact_id": "compiled-ext",
                "artifact_type": "compiled_extension",
                "name": "compiled",
                "source_kind": "s3",
                "source_uri": "s3://mirror/compiled",
                "immutable_ref": "sha256:" + ("4" * 64),
                "expected_digest": "sha256:" + ("4" * 64),
                "size_bytes": 100,
            },
        ],
    }


def _base_model(*, trust_remote_code: bool = False) -> dict[str, object]:
    return {
        "source": "hf://org/model",
        "tokenizer_revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "weights_revision": "cccccccccccccccccccccccccccccccccccccccc",
        "trust_remote_code": trust_remote_code,
    }


class _MirrorRequestHandler(http.server.BaseHTTPRequestHandler):
    root = Path(".")
    bearer_token: str | None = None

    def do_GET(self) -> None:  # noqa: N802
        if self.bearer_token is not None:
            auth = self.headers.get("Authorization")
            if auth != f"Bearer {self.bearer_token}":
                self.send_response(401)
                self.end_headers()
                return

        rel_path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path.lstrip("/"))
        target = (self.root / rel_path).resolve()
        try:
            target.relative_to(self.root.resolve())
        except ValueError:
            self.send_response(403)
            self.end_headers()
            return
        if not target.is_file():
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Length", str(target.stat().st_size))
        self.end_headers()
        self.wfile.write(target.read_bytes())

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@contextlib.contextmanager
def _serve_mirror(root: Path, *, bearer_token: str | None) -> str:
    handler = type(
        "MirrorRequestHandler",
        (_MirrorRequestHandler,),
        {
            "root": root,
            "bearer_token": bearer_token,
        },
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class TestHFResolution(unittest.TestCase):
    def test_resolve_hf_model_supports_nested_layouts_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = _nested_model_files()
            _write_files(root, files)
            client = FakeHFClient(root)

            resolved = resolve_hf_model(_base_model(), False, client=client, cache_dir=root)

            self.assertEqual(resolved.resolved_revision, client.commit)
            self.assertTrue(any(item["artifact_type"] == "model_config" for item in resolved.model_artifacts))
            self.assertTrue(any(item["artifact_type"] == "tokenizer" for item in resolved.model_artifacts))
            self.assertTrue(any(item["artifact_type"] == "generation_config" for item in resolved.model_artifacts))
            self.assertTrue(any(item["artifact_type"] == "chat_template" for item in resolved.model_artifacts))
            self.assertTrue(any(item["artifact_type"] == "prompt_formatter" for item in resolved.model_artifacts))
            self.assertEqual(
                len([item for item in resolved.model_artifacts if item["artifact_type"] == "model_weights"]),
                2,
            )
            self.assertTrue(all(item["expected_digest"].startswith("sha256:") for item in resolved.model_artifacts))

    def test_resolve_hf_model_with_remote_code_hashes_python_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = _nested_model_files()
            _write_files(root, files)
            client = FakeHFClient(root)

            resolved = resolve_hf_model(_base_model(trust_remote_code=True), True, client=client, cache_dir=root)

            self.assertTrue(any(item["artifact_type"] == "remote_code" for item in resolved.model_artifacts))
            rc_artifact = next(item for item in resolved.model_artifacts if item["artifact_type"] == "remote_code")
            self.assertEqual(rc_artifact["immutable_ref"], client.commit)
            self.assertEqual(rc_artifact["expected_digest"], _expected_remote_code_digest(files))

    def test_resolve_hf_model_offline_local_mirror_avoids_hf_client(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mirror_root = root / "mirror"
            files = _nested_model_files()
            _write_local_mirror(mirror_root, repo_id="org/model", commit=_COMMIT, files=files)
            client = FakeHFClient(root / "unused-client", fail_on_use=True)

            resolved = resolve_hf_model(
                _base_model(),
                False,
                client=client,
                cache_dir=root / "cache",
                mirror_root=mirror_root,
                resolution_mode="offline",
            )

            self.assertEqual(resolved.resolved_revision, _COMMIT)
            self.assertEqual(client.resolve_commit_calls, [])
            self.assertEqual(client.list_files_calls, [])
            self.assertEqual(client.download_calls, [])

    def test_resolve_hf_model_offline_http_mirror_uses_bearer_token(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mirror_root = root / "mirror"
            files = _nested_model_files()
            _write_local_mirror(mirror_root, repo_id="org/model", commit=_COMMIT, files=files)
            client = FakeHFClient(root / "unused-client", fail_on_use=True)

            with _serve_mirror(mirror_root, bearer_token="secret-token") as mirror_url:
                resolved = resolve_hf_model(
                    _base_model(),
                    False,
                    client=client,
                    cache_dir=root / "cache",
                    mirror_root=mirror_url,
                    resolution_mode="offline",
                    mirror_token="secret-token",
                )

            self.assertEqual(resolved.resolved_revision, _COMMIT)
            self.assertEqual(client.resolve_commit_calls, [])
            self.assertEqual(client.list_files_calls, [])
            self.assertEqual(client.download_calls, [])

    def test_resolve_hf_model_cache_first_falls_back_for_incomplete_mirror_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo_root = root / "repo"
            mirror_root = root / "mirror"
            files = _nested_model_files()
            _write_files(repo_root, files)
            _write_local_mirror(
                mirror_root,
                repo_id="org/model",
                commit=_COMMIT,
                files={
                    "configs/config.json": files["configs/config.json"],
                    "tokenizer/tokenizer.json": files["tokenizer/tokenizer.json"],
                    "templates/chat_template.jinja": files["templates/chat_template.jinja"],
                    "weights/model-00001.safetensors": files["weights/model-00001.safetensors"],
                },
            )
            client = FakeHFClient(repo_root)

            resolved = resolve_hf_model(
                _base_model(),
                False,
                client=client,
                cache_dir=root / "cache",
                mirror_root=mirror_root,
                resolution_mode="cache_first",
            )

            self.assertEqual(resolved.resolved_revision, _COMMIT)
            self.assertEqual(client.resolve_commit_calls, [])
            self.assertEqual(client.list_files_calls, [("org/model", _COMMIT)])
            self.assertIn(("org/model", _COMMIT, "settings/generation_config.json"), client.download_calls)
            self.assertIn(("org/model", _COMMIT, "formatters/prompt_formatter.py"), client.download_calls)

    def test_resolve_hf_model_rejects_missing_required_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = _nested_model_files()
            del files["settings/generation_config.json"]
            _write_files(root, files)
            client = FakeHFClient(root)

            with self.assertRaisesRegex(HFResolutionError, "generation_config"):
                resolve_hf_model(_base_model(), False, client=client, cache_dir=root)

    def test_resolve_hf_model_rejects_malformed_repo_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = _nested_model_files()
            _write_files(root, files)
            client = FakeHFClient(root, files=["../escape.py", *sorted(files.keys())])

            with self.assertRaisesRegex(HFResolutionError, "Invalid HF file path"):
                resolve_hf_model(_base_model(), False, client=client, cache_dir=root)

    def test_resolve_hf_model_rejects_remote_code_without_python_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = _no_python_model_files()
            _write_files(root, files)
            client = FakeHFClient(root)

            with self.assertRaisesRegex(HFResolutionError, "no python files found"):
                resolve_hf_model(_base_model(trust_remote_code=True), True, client=client, cache_dir=root)

    def test_resolver_merges_hf_artifacts_into_lockfile_with_offline_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mirror_root = root / "mirror"
            files = _nested_model_files()
            _write_local_mirror(mirror_root, repo_id="org/model", commit=_COMMIT, files=files)
            fake_client = FakeHFClient(root / "unused-client", fail_on_use=True)
            manifest = copy.deepcopy(_base_manifest())

            with mock.patch.object(resolver_main, "HuggingFaceHubClient", return_value=fake_client):
                lockfile = resolve_manifest_to_lockfile(
                    manifest,
                    resolve_hf=True,
                    hf_cache_dir=root / "cache",
                    hf_token=None,
                    hf_mirror_root=mirror_root,
                    hf_resolution_mode="offline",
                    hf_mirror_token=None,
                )

            model = manifest["model"]
            assert isinstance(model, dict)
            self.assertEqual(model["weights_revision"], _COMMIT)
            self.assertTrue(any(item["artifact_type"] == "model_weights" for item in lockfile["artifacts"]))
            expected_manifest_digest = "sha256:" + hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
            self.assertEqual(lockfile["manifest_digest"], expected_manifest_digest)


if __name__ == "__main__":
    unittest.main()
