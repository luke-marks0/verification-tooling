from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from typing import Any, Callable, Protocol

from modules.core.common.deterministic import sha256_file


class HFResolutionError(Exception):
    pass


class _MissingRequiredFilesError(HFResolutionError):
    pass


class HFClient(Protocol):
    def resolve_commit(self, repo_id: str, revision: str) -> str:
        ...

    def list_files(self, repo_id: str, revision: str) -> list[str]:
        ...

    def download_file(self, repo_id: str, revision: str, file_path: str, cache_dir: Path | None) -> Path:
        ...


class MirrorStore(Protocol):
    def resolve_commit(self, repo_id: str, revision: str) -> str:
        ...

    def list_files(self, repo_id: str, revision: str) -> list[str]:
        ...

    def download_file(self, repo_id: str, revision: str, file_path: str, cache_dir: Path | None) -> Path:
        ...


@dataclass(frozen=True)
class ResolvedHF:
    resolved_revision: str
    model_artifacts: list[dict[str, Any]]


HF_RESOLUTION_MODES = {"online", "cache_first", "offline"}
COMMIT_RE = re.compile(r"^[a-f0-9]{40}$")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth", ".pt")
CONFIG_CANDIDATES = ["config.json", "params.json"]
TOKENIZER_CANDIDATES = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "tokenization_config.json",
    "special_tokens_map.json",
]
GENERATION_CONFIG_CANDIDATES = ["generation_config.json", "generation_config.yaml", "generation_config.yml"]
CHAT_TEMPLATE_CANDIDATES = [
    "chat_template.jinja",
    "chat_template.json",
    "chat_template.txt",
    "tokenizer_config.json",
    "tokenization_config.json",
]
PROMPT_FORMATTER_CANDIDATES = [
    "prompt_formatter.py",
    "prompt_formatting.py",
    "format_prompt.py",
    "conversation.py",
    "chat_template.py",
    "processing.py",
    "preprocess.py",
    "tokenizer_config.json",
    "tokenization_config.json",
]


class HuggingFaceHubClient:
    def __init__(self, token: str | None = None) -> None:
        try:
            from huggingface_hub import HfApi
        except Exception as exc:  # pragma: no cover
            raise HFResolutionError(f"huggingface_hub import failed: {exc}")
        self._api = HfApi(token=token)

    def resolve_commit(self, repo_id: str, revision: str) -> str:
        info = self._api.model_info(repo_id=repo_id, revision=revision)
        sha = getattr(info, "sha", None)
        if not isinstance(sha, str) or not COMMIT_RE.fullmatch(sha):
            raise HFResolutionError(f"Unable to resolve immutable commit for {repo_id}@{revision}")
        return sha

    def list_files(self, repo_id: str, revision: str) -> list[str]:
        files = self._api.list_repo_files(repo_id=repo_id, revision=revision)
        if not isinstance(files, list):
            raise HFResolutionError(f"Unexpected list_repo_files response for {repo_id}@{revision}")
        return sorted(str(item) for item in files)

    def download_file(self, repo_id: str, revision: str, file_path: str, cache_dir: Path | None) -> Path:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # pragma: no cover
            raise HFResolutionError(f"huggingface_hub import failed: {exc}")

        target = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        return Path(target)


class LocalMirrorStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _repo_dir(self, repo_id: str) -> Path:
        return self.root.joinpath(*repo_id.split("/"))

    def _read_json(self, path: Path, *, description: str) -> Any:
        if not path.is_file():
            raise HFResolutionError(f"Mirror missing {description}: {path}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HFResolutionError(f"Mirror {description} is not valid JSON: {path}: {exc}") from exc

    def resolve_commit(self, repo_id: str, revision: str) -> str:
        if COMMIT_RE.fullmatch(revision):
            return revision
        refs = self._read_json(self._repo_dir(repo_id) / "refs.json", description="refs.json")
        if not isinstance(refs, dict):
            raise HFResolutionError(f"Mirror refs.json must be an object for {repo_id}")
        commit = refs.get(revision)
        if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
            raise HFResolutionError(f"Mirror missing immutable commit mapping for {repo_id}@{revision}")
        return commit

    def list_files(self, repo_id: str, revision: str) -> list[str]:
        files = self._read_json(
            self._repo_dir(repo_id) / "commits" / revision / "files.json",
            description=f"files.json for {repo_id}@{revision}",
        )
        if not isinstance(files, list) or not all(isinstance(item, str) and item.strip() for item in files):
            raise HFResolutionError(
                f"Mirror files.json must contain a non-empty list of file paths for {repo_id}@{revision}"
            )
        return sorted(str(item) for item in files)

    def download_file(self, repo_id: str, revision: str, file_path: str, cache_dir: Path | None) -> Path:
        del cache_dir
        safe_parts = _safe_relative_parts(file_path)
        target = self._repo_dir(repo_id) / "commits" / revision / "files"
        for part in safe_parts:
            target = target / part
        if not target.is_file():
            raise HFResolutionError(f"Mirror missing artifact for {repo_id}@{revision}: {file_path}")
        return target


class HTTPMirrorStore:
    def __init__(self, base_url: str, token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._scratch_root: Path | None = None

    def _headers(self) -> dict[str, str]:
        if self.token is None:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def _url(self, repo_id: str, *parts: str) -> str:
        encoded_parts = [urllib.parse.quote(part, safe="._-") for part in repo_id.split("/")]
        for part in parts:
            encoded_parts.extend(urllib.parse.quote(piece, safe="._-") for piece in _safe_relative_parts(part))
        return self.base_url + "/" + "/".join(encoded_parts)

    def _read_bytes(self, url: str, *, description: str) -> bytes:
        request = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(request) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise HFResolutionError(
                f"Mirror request failed for {description}: HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise HFResolutionError(
                f"Mirror request failed for {description}: {exc.reason}"
            ) from exc

    def _read_json(self, url: str, *, description: str) -> Any:
        raw = self._read_bytes(url, description=description)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HFResolutionError(f"Mirror {description} is not valid JSON: {url}: {exc}") from exc

    def _cache_root(self, cache_dir: Path | None) -> Path:
        if cache_dir is not None:
            root = cache_dir
        else:
            if self._scratch_root is None:
                self._scratch_root = Path(tempfile.mkdtemp(prefix="hf-mirror-cache-"))
            root = self._scratch_root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def resolve_commit(self, repo_id: str, revision: str) -> str:
        if COMMIT_RE.fullmatch(revision):
            return revision
        refs = self._read_json(self._url(repo_id, "refs.json"), description=f"refs.json for {repo_id}")
        if not isinstance(refs, dict):
            raise HFResolutionError(f"Mirror refs.json must be an object for {repo_id}")
        commit = refs.get(revision)
        if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
            raise HFResolutionError(f"Mirror missing immutable commit mapping for {repo_id}@{revision}")
        return commit

    def list_files(self, repo_id: str, revision: str) -> list[str]:
        files = self._read_json(
            self._url(repo_id, "commits", revision, "files.json"),
            description=f"files.json for {repo_id}@{revision}",
        )
        if not isinstance(files, list) or not all(isinstance(item, str) and item.strip() for item in files):
            raise HFResolutionError(
                f"Mirror files.json must contain a non-empty list of file paths for {repo_id}@{revision}"
            )
        return sorted(str(item) for item in files)

    def download_file(self, repo_id: str, revision: str, file_path: str, cache_dir: Path | None) -> Path:
        safe_parts = _safe_relative_parts(file_path)
        content = self._read_bytes(
            self._url(repo_id, "commits", revision, "files", file_path),
            description=f"{repo_id}@{revision}:{file_path}",
        )
        target = self._cache_root(cache_dir)
        for part in (*repo_id.split("/"), "commits", revision, "files", *safe_parts):
            target = target / part
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target


_MODEL_ROLE_TO_ARTIFACT_TYPE = {
    "weights_shard": "model_weights",
    "config": "model_config",
    "tokenizer": "tokenizer",
    "generation_config": "generation_config",
    "chat_template": "chat_template",
    "prompt_formatter": "prompt_formatter",
}


def parse_hf_source(source: str) -> str:
    match = re.fullmatch(r"hf://([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)", source)
    if not match:
        raise HFResolutionError(f"Invalid HF source: {source}")
    return match.group(1)


def _safe_relative_parts(file_path: str) -> tuple[str, ...]:
    pure = PurePosixPath(file_path)
    if pure.is_absolute() or len(pure.parts) == 0 or any(part in {"", ".", ".."} for part in pure.parts):
        raise HFResolutionError(f"Invalid HF file path: {file_path}")
    return tuple(str(part) for part in pure.parts)


def _normalize_repo_files(files: list[str]) -> list[str]:
    normalized: list[str] = []
    for item in files:
        if not isinstance(item, str) or item.strip() == "":
            raise HFResolutionError("HF repo file inventory must contain non-empty string paths")
        normalized.append("/".join(_safe_relative_parts(item)))
    return sorted(dict.fromkeys(normalized))


def _candidate_paths(files: list[str], basenames: list[str]) -> list[str]:
    priority = {name: idx for idx, name in enumerate(basenames)}
    matches = [path for path in files if PurePosixPath(path).name in priority]
    return sorted(
        matches,
        key=lambda path: (
            priority[PurePosixPath(path).name],
            len(PurePosixPath(path).parts),
            len(path),
            path,
        ),
    )


def _choose_preferred_path(files: list[str], basenames: list[str]) -> str | None:
    matches = _candidate_paths(files, basenames)
    if len(matches) == 0:
        return None
    return matches[0]


def _is_weight_file(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    if name.endswith(".index.json"):
        return False
    if not any(name.endswith(suffix) for suffix in WEIGHT_SUFFIXES):
        return False
    if "tokenizer" in name:
        return False
    return (
        any(token in name for token in ("model", "pytorch_model", "consolidated", "weights", "checkpoint", "params"))
        or name.endswith(".safetensors")
    )


def _select_required_paths(files: list[str]) -> dict[str, list[str] | str]:
    normalized = _normalize_repo_files(files)

    weights = [path for path in normalized if _is_weight_file(path)]
    config = _choose_preferred_path(normalized, CONFIG_CANDIDATES)
    tokenizer = _choose_preferred_path(normalized, TOKENIZER_CANDIDATES)
    generation_config = _choose_preferred_path(normalized, GENERATION_CONFIG_CANDIDATES)
    chat_template = _choose_preferred_path(normalized, CHAT_TEMPLATE_CANDIDATES)
    prompt_formatter = _choose_preferred_path(normalized, PROMPT_FORMATTER_CANDIDATES)

    missing: list[str] = []
    if len(weights) == 0:
        missing.append("weights_shard")
    if config is None:
        missing.append("config")
    if tokenizer is None:
        missing.append("tokenizer")
    if generation_config is None:
        missing.append("generation_config")
    if chat_template is None:
        missing.append("chat_template")
    if prompt_formatter is None:
        missing.append("prompt_formatter")

    if missing:
        raise _MissingRequiredFilesError("Required model files missing from HF repo: " + ", ".join(missing))

    return {
        "weights_shard": weights,
        "config": config,
        "tokenizer": tokenizer,
        "generation_config": generation_config,
        "chat_template": chat_template,
        "prompt_formatter": prompt_formatter,
    }


def _remote_code_digest(
    *,
    python_files: list[str],
    download_file: Callable[[str], Path],
) -> str:
    h = hashlib.sha256()
    for file_path in sorted(python_files):
        h.update(file_path.encode("utf-8"))
        h.update(b"\0")
        local = download_file(file_path)
        h.update(local.read_bytes())
        h.update(b"\0")
    return "sha256:" + h.hexdigest()


def _artifact_id(role: str, path: str) -> str:
    name = Path(path).name
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    return f"hf-{role}-{safe_name}"


def _mirror_store_from_root(mirror_root: str | Path, mirror_token: str | None) -> MirrorStore:
    if isinstance(mirror_root, Path):
        return LocalMirrorStore(mirror_root)
    parsed = urllib.parse.urlparse(mirror_root)
    if parsed.scheme in {"http", "https"}:
        return HTTPMirrorStore(mirror_root, token=mirror_token)
    return LocalMirrorStore(Path(mirror_root))


def _resolve_commit(
    *,
    repo_id: str,
    requested_revision: str,
    resolved_revision_hint: str | None,
    client: HFClient,
    mirror: MirrorStore | None,
    resolution_mode: str,
) -> str:
    if COMMIT_RE.fullmatch(requested_revision):
        return requested_revision

    if mirror is not None and resolution_mode in {"cache_first", "offline"}:
        try:
            return mirror.resolve_commit(repo_id, requested_revision)
        except HFResolutionError:
            if resolution_mode == "offline":
                if isinstance(resolved_revision_hint, str) and COMMIT_RE.fullmatch(resolved_revision_hint):
                    return resolved_revision_hint
                raise

    if resolution_mode == "offline":
        raise HFResolutionError(
            f"Offline HF resolution requires mirror metadata or manifest.resolved_revision commit for {repo_id}@{requested_revision}"
        )
    return client.resolve_commit(repo_id, requested_revision)


def _resolve_file_list(
    *,
    repo_id: str,
    revision: str,
    client: HFClient,
    mirror: MirrorStore | None,
    resolution_mode: str,
) -> list[str]:
    if mirror is not None and resolution_mode in {"cache_first", "offline"}:
        try:
            return mirror.list_files(repo_id, revision)
        except HFResolutionError:
            if resolution_mode == "offline":
                raise
    if resolution_mode == "offline":
        raise HFResolutionError(f"Offline HF resolution requires mirror file inventory for {repo_id}@{revision}")
    return client.list_files(repo_id, revision)


def _download_resolved_file(
    *,
    repo_id: str,
    revision: str,
    file_path: str,
    client: HFClient,
    mirror: MirrorStore | None,
    resolution_mode: str,
    cache_dir: Path | None,
) -> Path:
    if mirror is not None and resolution_mode in {"cache_first", "offline"}:
        try:
            return mirror.download_file(repo_id, revision, file_path, cache_dir)
        except HFResolutionError:
            if resolution_mode == "offline":
                raise
    if resolution_mode == "offline":
        raise HFResolutionError(f"Offline HF resolution requires mirrored artifact for {repo_id}@{revision}: {file_path}")
    return client.download_file(repo_id, revision, file_path, cache_dir)


def _resolve_required_inventory(
    *,
    repo_id: str,
    revision: str,
    client: HFClient,
    mirror: MirrorStore | None,
    resolution_mode: str,
) -> tuple[list[str], dict[str, list[str] | str]]:
    files = _resolve_file_list(
        repo_id=repo_id,
        revision=revision,
        client=client,
        mirror=mirror,
        resolution_mode=resolution_mode,
    )
    try:
        return files, _select_required_paths(files)
    except _MissingRequiredFilesError:
        if mirror is None or resolution_mode != "cache_first":
            raise
        client_files = client.list_files(repo_id, revision)
        combined = sorted(dict.fromkeys([*files, *client_files]))
        return combined, _select_required_paths(combined)


def resolve_hf_model(
    model: dict[str, Any],
    trust_remote_code: bool,
    *,
    client: HFClient,
    cache_dir: Path | None = None,
    mirror_root: str | Path | None = None,
    resolution_mode: str = "online",
    mirror_token: str | None = None,
) -> ResolvedHF:
    if resolution_mode not in HF_RESOLUTION_MODES:
        raise HFResolutionError(f"Unsupported HF resolution_mode: {resolution_mode}")
    if resolution_mode != "online" and mirror_root is None:
        raise HFResolutionError(f"HF resolution_mode={resolution_mode} requires mirror_root")

    repo_id = parse_hf_source(model["source"])
    requested_revision = model.get("requested_revision") or model.get("resolved_revision") or "main"
    if not isinstance(requested_revision, str) or requested_revision.strip() == "":
        raise HFResolutionError("Model requested/resolved revision must be non-empty")

    mirror = _mirror_store_from_root(mirror_root, mirror_token) if mirror_root is not None else None
    commit = _resolve_commit(
        repo_id=repo_id,
        requested_revision=requested_revision,
        resolved_revision_hint=model.get("resolved_revision"),
        client=client,
        mirror=mirror,
        resolution_mode=resolution_mode,
    )
    files, selected = _resolve_required_inventory(
        repo_id=repo_id,
        revision=commit,
        client=client,
        mirror=mirror,
        resolution_mode=resolution_mode,
    )

    artifacts: list[dict[str, Any]] = []
    downloaded: dict[str, Path] = {}

    def materialize(file_path: str) -> Path:
        normalized_path = "/".join(_safe_relative_parts(file_path))
        if normalized_path not in downloaded:
            downloaded[normalized_path] = _download_resolved_file(
                repo_id=repo_id,
                revision=commit,
                file_path=normalized_path,
                client=client,
                mirror=mirror,
                resolution_mode=resolution_mode,
                cache_dir=cache_dir,
            )
        return downloaded[normalized_path]

    def add_file(role: str, file_path: str) -> None:
        local = materialize(file_path)
        digest = sha256_file(local)
        size_bytes = local.stat().st_size
        uri = f"hf://{repo_id}/{file_path}"

        artifacts.append(
            {
                "artifact_id": _artifact_id(role, file_path),
                "artifact_type": _MODEL_ROLE_TO_ARTIFACT_TYPE[role],
                "name": Path(file_path).name,
                "source_kind": "hf",
                "source_uri": uri,
                "immutable_ref": commit,
                "expected_digest": digest,
                "size_bytes": size_bytes,
                "path": file_path,
                "role": role,
            }
        )

    for weights_path in selected["weights_shard"]:  # type: ignore[index]
        add_file("weights_shard", str(weights_path))

    for role in ["config", "tokenizer", "generation_config", "chat_template", "prompt_formatter"]:
        add_file(role, str(selected[role]))  # type: ignore[index]

    if trust_remote_code:
        python_files = [item for item in _normalize_repo_files(files) if item.endswith(".py")]
        if not python_files:
            raise HFResolutionError("trust_remote_code=true but no python files found in repository")
        rc_digest = _remote_code_digest(
            python_files=python_files,
            download_file=materialize,
        )
        artifacts.append(
            {
                "artifact_id": "hf-remote-code",
                "artifact_type": "remote_code",
                "name": "remote_code",
                "source_kind": "hf",
                "source_uri": f"hf://{repo_id}?revision={commit}#remote_code",
                "immutable_ref": commit,
                "expected_digest": rc_digest,
                "size_bytes": max(1, len(python_files)),
            }
        )

    artifacts = sorted(artifacts, key=lambda item: (item["artifact_type"], item["artifact_id"], item["expected_digest"]))

    return ResolvedHF(
        resolved_revision=commit,
        model_artifacts=artifacts,
    )
