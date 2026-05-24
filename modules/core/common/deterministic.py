from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json_text(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def canonical_json_bytes(data: Any) -> bytes:
    return canonical_json_text(data).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_prefixed(data: bytes) -> str:
    return f"sha256:{sha256_hex(data)}"


def sha256_prefixed_text(text: str) -> str:
    return sha256_prefixed(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_prefixed(path.read_bytes())


def flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(flatten_numbers(item))
        return out
    return []


def first_mismatch_path(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return path

    if isinstance(left, dict):
        left_keys = sorted(left.keys())
        right_keys = sorted(right.keys())
        if left_keys != right_keys:
            return path
        for key in left_keys:
            next_path = f"{path}.{key}"
            mismatch = first_mismatch_path(left[key], right[key], next_path)
            if mismatch is not None:
                return mismatch
        return None

    if isinstance(left, list):
        if len(left) != len(right):
            return path
        for idx, (lval, rval) in enumerate(zip(left, right)):
            mismatch = first_mismatch_path(lval, rval, f"{path}[{idx}]")
            if mismatch is not None:
                return mismatch
        return None

    if left != right:
        return path
    return None


def stable_sort_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        artifacts,
        key=lambda a: (
            str(a.get("artifact_type", "")),
            str(a.get("artifact_id", "")),
            str(a.get("digest", "")),
            str(a.get("immutable_ref", "")),
        ),
    )


def lockfile_for_digest(lockfile: dict[str, Any]) -> dict[str, Any]:
    clone = deepcopy(lockfile)
    canonicalization = clone.setdefault("canonicalization", {})
    canonicalization["lockfile_digest"] = "sha256:" + ("0" * 64)
    return clone


def compute_lockfile_digest(lockfile: dict[str, Any]) -> str:
    return sha256_prefixed(canonical_json_bytes(lockfile_for_digest(lockfile)))


def compute_bundle_digest(bundle: dict[str, Any]) -> str:
    clone = deepcopy(bundle)
    clone["bundle_digest"] = "sha256:" + ("0" * 64)
    return sha256_prefixed(canonical_json_bytes(clone))
