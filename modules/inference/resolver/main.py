#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.core.common.contracts import ValidationError, validate_with_schema
from modules.core.common.deterministic import (
    canonical_json_bytes,
    canonical_json_text,
    compute_lockfile_digest,
    sha256_prefixed,
    stable_sort_artifacts,
)
from modules.core.common.hf_resolution import (
    HFResolutionError,
    HuggingFaceHubClient,
    resolve_hf_model,
)
from modules.inference.manifest.model import Manifest

MODEL_ARTIFACT_TYPES = {
    "model_weights",
    "model_config",
    "tokenizer",
    "generation_config",
    "chat_template",
    "prompt_formatter",
    "remote_code",
}


def _artifact_from_input(item: dict[str, Any], model_source: str) -> dict[str, Any]:
    source_bytes = canonical_json_bytes(item)
    digest = item.get("expected_digest") or sha256_prefixed(source_bytes)
    return {
        "artifact_id": item["artifact_id"],
        "artifact_type": item["artifact_type"],
        "name": item.get("name", item["artifact_id"]),
        "source_kind": item["source_kind"],
        "uri": item["source_uri"],
        "immutable_ref": item["immutable_ref"],
        "digest": digest,
        "size_bytes": int(item.get("size_bytes", max(1, len(source_bytes)))),
        "hash_algorithm": "sha256",
        "resolved_from": model_source,
        "build_output": item["artifact_type"] in {"compiled_extension", "kernel_library"},
    }


def _merge_model_artifacts(
    existing_artifact_inputs: list[dict[str, Any]],
    resolved_model_artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keep = [item for item in existing_artifact_inputs if item.get("artifact_type") not in MODEL_ARTIFACT_TYPES]
    merged = [*keep, *resolved_model_artifacts]
    merged = sorted(
        merged,
        key=lambda item: (
            str(item.get("artifact_type", "")),
            str(item.get("artifact_id", "")),
            str(item.get("immutable_ref", "")),
            str(item.get("expected_digest", "")),
        ),
    )
    return merged


def _resolve_manifest_hf_model(
    manifest: dict[str, Any],
    *,
    hf_cache_dir: Path | None,
    hf_token: str | None,
    hf_mirror_root: str | Path | None,
    hf_resolution_mode: str,
    hf_mirror_token: str | None,
) -> dict[str, Any]:
    source = manifest["model"]["source"]
    if not str(source).startswith("hf://"):
        return manifest

    client = HuggingFaceHubClient(token=hf_token)
    resolved = resolve_hf_model(
        manifest["model"],
        bool(manifest["model"]["trust_remote_code"]),
        client=client,
        cache_dir=hf_cache_dir,
        mirror_root=hf_mirror_root,
        resolution_mode=hf_resolution_mode,
        mirror_token=hf_mirror_token,
    )

    manifest["model"]["weights_revision"] = resolved.resolved_revision
    manifest["model"]["tokenizer_revision"] = resolved.resolved_revision

    manifest["artifact_inputs"] = _merge_model_artifacts(manifest["artifact_inputs"], resolved.model_artifacts)
    return manifest


def resolve_manifest_to_lockfile(
    manifest: dict[str, Any],
    *,
    resolve_hf: bool = False,
    hf_cache_dir: Path | None = None,
    hf_token: str | None = None,
    hf_mirror_root: str | Path | None = None,
    hf_resolution_mode: str = "online",
    hf_mirror_token: str | None = None,
) -> dict[str, Any]:
    validate_with_schema("manifest.v1.schema.json", manifest)
    if resolve_hf:
        manifest = _resolve_manifest_hf_model(
            manifest,
            hf_cache_dir=hf_cache_dir,
            hf_token=hf_token,
            hf_mirror_root=hf_mirror_root,
            hf_resolution_mode=hf_resolution_mode,
            hf_mirror_token=hf_mirror_token,
        )
        validate_with_schema("manifest.v1.schema.json", manifest)
    # Validate the (possibly mutated) manifest with Pydantic before lockfile generation.
    Manifest.model_validate(manifest)

    deterministic_timestamp = manifest["created_at"]

    artifacts = [_artifact_from_input(item, manifest["model"]["source"]) for item in manifest["artifact_inputs"]]
    artifacts = stable_sort_artifacts(artifacts)

    runtime_seed = {
        "runtime": manifest["runtime"],
        "hardware": manifest["hardware_profile"],
    }

    lockfile = {
        "lockfile_version": "v1",
        "generated_at": deterministic_timestamp,
        "manifest_digest": sha256_prefixed(canonical_json_bytes(manifest)),
        "runtime_closure_digest": sha256_prefixed(canonical_json_bytes(runtime_seed)),
        "resolver": {
            "name": "deterministic-resolver",
            "version": "0.1.0"
        },
        "canonicalization": {
            "method": "json_canonical_v1",
            "lockfile_digest": "sha256:" + ("0" * 64)
        },
        "artifacts": artifacts,
        "attestations": [
            {
                "attestation_type": "resolver_provenance",
                "signer": "resolver@deterministic-serving-stack",
                "statement_digest": sha256_prefixed(canonical_json_bytes({"artifacts": artifacts})),
                "timestamp": deterministic_timestamp,
            }
        ],
    }
    lockfile["canonicalization"]["lockfile_digest"] = compute_lockfile_digest(lockfile)
    validate_with_schema("lockfile.v1.schema.json", lockfile)
    return lockfile


def _read_optional_secret(value: str | None, path: str | None) -> str | None:
    if value is not None:
        return value
    if path is None:
        return None
    try:
        secret = Path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise HFResolutionError(f"Unable to read secret file {path}: {exc}") from exc
    if secret == "":
        raise HFResolutionError(f"Secret file {path} is empty")
    return secret


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve manifest into deterministic lockfile")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--lockfile-out", required=True, help="Path to lockfile JSON output")
    parser.add_argument("--manifest-out", help="Path to write the resolved manifest (after HF resolution)")
    parser.add_argument("--resolve-hf", action="store_true", help="Resolve Hugging Face model files and digests")
    parser.add_argument("--hf-cache-dir", help="Optional HF cache directory")
    hf_token_group = parser.add_mutually_exclusive_group()
    hf_token_group.add_argument("--hf-token", help="Optional HF token")
    hf_token_group.add_argument("--hf-token-file", help="Read the HF token from a file")
    parser.add_argument(
        "--hf-resolution-mode",
        choices=["online", "cache_first", "offline"],
        default="online",
        help="How HF resolution should use the internal mirror/cache",
    )
    parser.add_argument("--hf-mirror-root", help="Optional local path or HTTP(S) base URL for an HF mirror")
    hf_mirror_token_group = parser.add_mutually_exclusive_group()
    hf_mirror_token_group.add_argument("--hf-mirror-token", help="Optional bearer token for an HTTP HF mirror")
    hf_mirror_token_group.add_argument("--hf-mirror-token-file", help="Read the HF mirror bearer token from a file")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    lockfile_path = Path(args.lockfile_out)
    hf_token = _read_optional_secret(args.hf_token, args.hf_token_file)
    hf_mirror_token = _read_optional_secret(args.hf_mirror_token, args.hf_mirror_token_file)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lockfile = resolve_manifest_to_lockfile(
        manifest,
        resolve_hf=bool(args.resolve_hf),
        hf_cache_dir=Path(args.hf_cache_dir) if args.hf_cache_dir else None,
        hf_token=hf_token,
        hf_mirror_root=args.hf_mirror_root,
        hf_resolution_mode=args.hf_resolution_mode,
        hf_mirror_token=hf_mirror_token,
    )

    lockfile_path.parent.mkdir(parents=True, exist_ok=True)
    lockfile_path.write_text(canonical_json_text(lockfile), encoding="utf-8")

    if args.manifest_out:
        manifest_out_path = Path(args.manifest_out)
        manifest_out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_out_path.write_text(canonical_json_text(manifest), encoding="utf-8")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HFResolutionError as exc:
        print(str(exc))
        raise SystemExit(1)
    except ValidationError as exc:
        print(str(exc))
        raise SystemExit(1)
