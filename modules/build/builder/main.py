#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
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


CLOSURE_COMPONENT_RULES: tuple[tuple[str, set[str]], ...] = (
    ("serving_stack", {"serving_stack"}),
    ("cuda_userspace_or_container", {"cuda_lib", "container_image"}),
    ("kernel_libraries", {"kernel_library", "compiled_extension"}),
)

SHA256_RE = re.compile(r"^sha256:[a-f0-9]{64}$")


def _component_artifacts(
    artifacts: list[dict[str, Any]],
    allowed_types: set[str],
) -> list[dict[str, Any]]:
    selected = [item for item in artifacts if item["artifact_type"] in allowed_types]
    return sorted(
        selected,
        key=lambda item: (
            str(item["artifact_type"]),
            str(item["artifact_id"]),
            str(item["digest"]),
            str(item["immutable_ref"]),
        ),
    )


def _component_descriptor(name: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
    digest_seed = [
        {
            "artifact_id": item["artifact_id"],
            "artifact_type": item["artifact_type"],
            "digest": item["digest"],
            "immutable_ref": item["immutable_ref"],
            "uri": item["uri"],
        }
        for item in selected
    ]
    return {
        "name": name,
        "artifact_types": sorted({item["artifact_type"] for item in selected}),
        "artifact_ids": [item["artifact_id"] for item in selected],
        "artifact_count": len(selected),
        "artifact_digest": sha256_prefixed(canonical_json_bytes(digest_seed)),
    }


def _collect_closure_components(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # The software stack is pinned by the Nix runtime closure
    # (``runtime_closure_digest`` / ``build.nix_closure``), not by per-component
    # manifest artifacts. If a manifest happens to enumerate them they're
    # recorded here as a bill of materials, but absence is not an error.
    components: list[dict[str, Any]] = []
    for name, allowed_types in CLOSURE_COMPONENT_RULES:
        selected = _component_artifacts(artifacts, allowed_types)
        if len(selected) == 0:
            continue
        components.append(_component_descriptor(name, selected))
    return sorted(components, key=lambda item: str(item["name"]))


def _collect_oci_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in artifacts:
        if item["source_kind"] != "oci":
            continue
        immutable_ref = str(item.get("immutable_ref", ""))
        uri = str(item.get("uri", ""))
        if not (SHA256_RE.match(immutable_ref) or "@sha256:" in uri):
            continue
        out.append(
            {
                "artifact_id": item["artifact_id"],
                "artifact_type": item["artifact_type"],
                "uri": uri,
                "digest": item["digest"],
                "immutable_ref": immutable_ref,
            }
        )
    return sorted(out, key=lambda item: (str(item["artifact_type"]), str(item["artifact_id"]), str(item["digest"])))


def _collect_collective_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [item for item in artifacts if item["artifact_type"] == "collective_stack"]
    out = [
        {
            "artifact_id": item["artifact_id"],
            "digest": item["digest"],
            "immutable_ref": item["immutable_ref"],
            "uri": item["uri"],
        }
        for item in selected
    ]
    return sorted(out, key=lambda item: (str(item["artifact_id"]), str(item["digest"])))


def _reference_nix_closure(
    artifacts: list[dict[str, Any]],
    closure_digest: str,
    *,
    source: str,
) -> dict[str, Any]:
    digest_suffix = closure_digest.split(":", 1)[1]
    store_prefix = "/nix/store" if source != "equivalent_descriptor" else "/equivalent/store"
    return {
        "source": source,
        "store_paths": [f"{store_prefix}/{digest_suffix[:32]}-runtime-closure"],
        "derivation_paths": [],
        "closure_size_bytes": sum(int(item["size_bytes"]) for item in artifacts),
        "closure_digest": closure_digest,
    }


def _normalize_nix_path_entries(raw: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    if isinstance(raw, dict):
        iterable = [{"path": path, **info} for path, info in raw.items() if isinstance(info, dict)]
    elif isinstance(raw, list):
        iterable = raw
    else:
        raise ValidationError("Unexpected nix path-info JSON structure")

    for item in iterable:
        if not isinstance(item, dict):
            raise ValidationError("nix path-info returned a non-object entry")
        path = item.get("path") or item.get("storePath")
        if not isinstance(path, str) or path.strip() == "":
            raise ValidationError("nix path-info entry missing path")

        references = item.get("references", [])
        if not isinstance(references, list):
            references = []
        deriver = item.get("deriver")
        nar_size = item.get("narSize", 0)
        if not isinstance(nar_size, int):
            try:
                nar_size = int(nar_size)
            except Exception as exc:
                raise ValidationError(f"Invalid nix narSize for {path}: {nar_size}") from exc

        entries.append(
            {
                "path": path,
                "nar_hash": str(item.get("narHash", "")),
                "nar_size": max(0, nar_size),
                "references": sorted(str(ref) for ref in references),
                "deriver": str(deriver) if isinstance(deriver, str) and deriver else "",
            }
        )

    entries = sorted(entries, key=lambda item: str(item["path"]))
    if len(entries) == 0:
        raise ValidationError("nix path-info returned no closure entries")
    return entries


def _query_nix_closure(nix_store_paths: list[str]) -> dict[str, Any]:
    if shutil.which("nix") is None:
        raise ValidationError("nix executable not found but --nix-store-path was provided")

    cmd = ["nix", "path-info", "--json", "--recursive", *nix_store_paths]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise ValidationError(f"nix path-info failed: {detail or exc}") from exc

    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"nix path-info output was not valid JSON: {exc}") from exc

    entries = _normalize_nix_path_entries(raw)
    derivations = sorted({item["deriver"] for item in entries if item["deriver"]})
    closure_digest = sha256_prefixed(canonical_json_bytes(entries))
    closure_size_bytes = sum(int(item["nar_size"]) for item in entries)
    return {
        "source": "nix_cli",
        "store_paths": [item["path"] for item in entries],
        "derivation_paths": derivations,
        "closure_size_bytes": max(1, closure_size_bytes),
        "closure_digest": closure_digest,
    }


def _oci_image_descriptor(oci_artifacts: list[dict[str, Any]]) -> dict[str, Any] | None:
    # Optional: only present when the manifest enumerates OCI artifacts. The
    # authoritative image is the Nix-built closure (``build.nix_closure``).
    if len(oci_artifacts) == 0:
        return None

    preferred = next(
        (item for item in oci_artifacts if item["artifact_type"] == "serving_stack"),
        oci_artifacts[0],
    )
    return {
        "image_ref": preferred["uri"],
        "image_digest": preferred["immutable_ref"] if SHA256_RE.match(preferred["immutable_ref"]) else preferred["digest"],
        "source_artifact_id": preferred["artifact_id"],
    }


def _build_seed(
    *,
    builder_system: str,
    resolver: dict[str, Any],
    components: list[dict[str, Any]],
    oci_artifacts: list[dict[str, Any]],
    collective_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "builder_system": builder_system,
        "resolver": resolver,
        "components": components,
        "oci_artifacts": oci_artifacts,
        "collective_artifacts": collective_artifacts,
    }


def _attestation_statement(
    *,
    builder_system: str,
    closure_uri: str,
    closure_inputs_digest: str,
    components: list[dict[str, Any]],
    oci_artifacts: list[dict[str, Any]],
    oci_image: dict[str, Any] | None,
    nix_closure: dict[str, Any],
    collective_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    statement: dict[str, Any] = {
        "builder_system": builder_system,
        "closure_uri": closure_uri,
        "closure_inputs_digest": closure_inputs_digest,
        "components": [
            {
                "name": item["name"],
                "artifact_count": item["artifact_count"],
                "artifact_digest": item["artifact_digest"],
            }
            for item in components
        ],
        "oci_artifacts": [
            {
                "artifact_id": item["artifact_id"],
                "digest": item["digest"],
            }
            for item in oci_artifacts
        ],
        "nix_closure": nix_closure,
        "collective_stack_artifacts": collective_artifacts,
    }
    if oci_image is not None:
        statement["oci_image"] = oci_image
    return statement


def build_runtime(
    lockfile: dict[str, Any],
    *,
    builder_system: str = "nix",
    nix_store_paths: list[str] | None = None,
    closure_digest: str | None = None,
) -> dict[str, Any]:
    if builder_system not in {"nix", "equivalent"}:
        raise ValidationError(f"Unsupported builder_system: {builder_system}")

    validate_with_schema("lockfile.v1.schema.json", lockfile)
    expected = compute_lockfile_digest(lockfile)
    actual = lockfile["canonicalization"]["lockfile_digest"]
    if expected != actual:
        raise ValidationError(
            f"Input lockfile canonicalization.lockfile_digest mismatch: expected={expected} actual={actual}"
        )

    artifacts = stable_sort_artifacts(lockfile["artifacts"])
    deterministic_timestamp = lockfile["generated_at"]
    components = _collect_closure_components(artifacts)
    oci_artifacts = _collect_oci_artifacts(artifacts)
    collective_artifacts = _collect_collective_artifacts(artifacts)
    closure_seed = _build_seed(
        builder_system=builder_system,
        resolver=lockfile["resolver"],
        components=components,
        oci_artifacts=oci_artifacts,
        collective_artifacts=collective_artifacts,
    )
    reference_digest = sha256_prefixed(canonical_json_bytes(closure_seed))
    if nix_store_paths:
        nix_closure = _query_nix_closure(nix_store_paths)
    elif closure_digest:
        # Pre-computed closure digest (e.g. from a prior `nix path-info` run)
        nix_closure = _reference_nix_closure(
            artifacts,
            closure_digest,
            source="nix_cli",
        )
    else:
        nix_closure = _reference_nix_closure(
            artifacts,
            reference_digest,
            source="reference_descriptor" if builder_system == "nix" else "equivalent_descriptor",
        )

    closure_inputs_digest = str(nix_closure["closure_digest"])
    closure_uri = f"{builder_system}://closure/{closure_inputs_digest.split(':', 1)[1]}"
    oci_image = _oci_image_descriptor(oci_artifacts)

    lockfile["artifacts"] = artifacts
    lockfile["runtime_closure_digest"] = closure_inputs_digest
    lockfile["generated_at"] = deterministic_timestamp
    build_section: dict[str, Any] = {
        "builder_system": builder_system,
        "closure_uri": closure_uri,
        "closure_inputs_digest": closure_inputs_digest,
        "components": components,
        "oci_artifacts": oci_artifacts,
        "nix_closure": nix_closure,
        "collective_stack_artifacts": collective_artifacts,
    }
    if oci_image is not None:
        build_section["oci_image"] = oci_image
    lockfile["build"] = build_section

    statement = _attestation_statement(
        builder_system=builder_system,
        closure_uri=closure_uri,
        closure_inputs_digest=closure_inputs_digest,
        components=components,
        oci_artifacts=oci_artifacts,
        oci_image=oci_image,
        nix_closure=nix_closure,
        collective_artifacts=collective_artifacts,
    )
    attestations = [item for item in lockfile.get("attestations", []) if item.get("attestation_type") != "build_provenance"]
    attestations.append(
        {
            "attestation_type": "build_provenance",
            "signer": "builder@deterministic-serving-stack",
            "statement_digest": sha256_prefixed(canonical_json_bytes(statement)),
            "timestamp": deterministic_timestamp,
        }
    )
    lockfile["attestations"] = attestations

    lockfile["canonicalization"]["method"] = "json_canonical_v1"
    lockfile["canonicalization"]["lockfile_digest"] = compute_lockfile_digest(lockfile)
    validate_with_schema("lockfile.v1.schema.json", lockfile)
    return lockfile


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic runtime closure digest")
    parser.add_argument("--lockfile", required=True, help="Input lockfile")
    parser.add_argument("--lockfile-out", required=True, help="Output lockfile")
    parser.add_argument("--builder-system", default="nix", choices=["nix", "equivalent"], help="Hermetic builder system")
    parser.add_argument(
        "--nix-store-path",
        action="append",
        default=[],
        help="Optional Nix store path to query via nix path-info for closure metadata",
    )
    parser.add_argument(
        "--closure-digest",
        help="Pre-computed runtime closure digest (sha256:...) from a prior Nix build",
    )
    args = parser.parse_args()

    lockfile_path = Path(args.lockfile)
    out_path = Path(args.lockfile_out)

    lockfile = json.loads(lockfile_path.read_text(encoding="utf-8"))
    built = build_runtime(
        lockfile,
        builder_system=args.builder_system,
        nix_store_paths=[str(item) for item in args.nix_store_path],
        closure_digest=args.closure_digest,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(canonical_json_text(built), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(str(exc))
        raise SystemExit(1)
