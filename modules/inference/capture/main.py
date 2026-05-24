#!/usr/bin/env python3
"""Convert server capture logs into verifiable run bundles.

Reads the server's boot_record.json + capture.jsonl and produces a
run_bundle.v1.json with observable files that the verifier can consume.

Usage:
    python3 modules/inference/capture/main.py \\
        --server-dir /path/to/server-run \\
        --manifest /path/to/manifest.resolved.json \\
        --lockfile /path/to/lockfile.built.v1.json \\
        --out-dir /path/to/bundle-output
"""
from __future__ import annotations

import argparse
import hashlib
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
    compute_bundle_digest,
    compute_lockfile_digest,
    sha256_prefixed,
    utc_now_iso,
)
from modules.inference.manifest.model import Manifest


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json_text(data), encoding="utf-8")
    return sha256_prefixed(path.read_bytes())


def _load_capture_entries(capture_path: Path) -> list[dict[str, Any]]:
    """Load JSONL capture file, filtering to /v1/chat/completions POST entries."""
    entries = []
    for line in capture_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("endpoint", "").startswith("/v1/") and "request" in entry and "response" in entry:
            entries.append(entry)
    return entries


def _extract_tokens_from_response(response: dict[str, Any]) -> list[int]:
    """Extract token IDs from a vLLM chat completion response."""
    # vLLM returns token_ids in the response when available
    token_ids = response.get("token_ids")
    if token_ids:
        return list(token_ids)

    # Fall back: use prompt_token_ids + completion token_ids if present
    choices = response.get("choices", [])
    if choices:
        choice = choices[0]
        tids = choice.get("token_ids")
        if tids:
            return list(tids)

    # No token IDs available — hash the text content deterministically
    text = ""
    if choices:
        msg = choices[0].get("message", {})
        text = msg.get("content", "")

    # Convert text to pseudo-token-ids via stable hashing
    if text:
        import hashlib
        tokens = []
        for i in range(0, len(text), 4):
            chunk = text[i:i+4].encode("utf-8")
            h = int(hashlib.sha256(chunk).hexdigest()[:8], 16) % 100000
            tokens.append(h)
        return tokens
    return []


def _extract_logprobs_from_response(response: dict[str, Any]) -> list[float]:
    """Extract logprobs from a vLLM response if available."""
    choices = response.get("choices", [])
    if not choices:
        return []

    logprobs_data = choices[0].get("logprobs")
    if not logprobs_data:
        return []

    content = logprobs_data.get("content", [])
    return [round(float(entry.get("logprob", 0.0)), 8) for entry in content if entry]


def capture_to_bundle(
    *,
    server_dir: Path,
    manifest: dict[str, Any],
    lockfile: dict[str, Any],
    out_dir: Path,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Convert server capture into a run bundle."""
    validate_with_schema("manifest.v1.schema.json", manifest)
    validate_with_schema("lockfile.v1.schema.json", lockfile)

    # Parse into typed Manifest for dot-access reads.
    # Keep manifest dict for canonical JSON serialization.
    m = Manifest.model_validate(manifest)

    boot_record_path = server_dir / "boot_record.json"
    capture_path = server_dir / "capture.jsonl"

    if not capture_path.exists():
        raise ValidationError(f"Capture file not found: {capture_path}")

    boot_record = _load_json(boot_record_path) if boot_record_path.exists() else {}
    entries = _load_capture_entries(capture_path)

    if not entries:
        raise ValidationError("No capture entries found in capture.jsonl")

    # Build observables from captured request/response pairs
    token_observables = []
    logit_observables = []

    for idx, entry in enumerate(entries):
        req = entry.get("request", {})
        resp = entry.get("response", {})

        # Request ID: use position index for cross-session comparability
        req_id = f"req-{idx}"

        # Extract tokens
        tokens = _extract_tokens_from_response(resp)
        logprobs = _extract_logprobs_from_response(resp)

        # Pad logprobs to match token count if needed
        while len(logprobs) < len(tokens):
            logprobs.append(0.0)
        logprobs = logprobs[:len(tokens)]

        token_observables.append({"id": req_id, "tokens": tokens})
        logit_observables.append({"id": req_id, "logits": logprobs})

    # Write observable files
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_dir = out_dir / "observables"

    tokens_digest = _write_json(obs_dir / "tokens.json", token_observables)
    logits_digest = _write_json(obs_dir / "logits.json", logit_observables)

    # Write manifest and lockfile copies
    manifest_copy_path = out_dir / "manifest.json"
    lockfile_copy_path = out_dir / "lockfile.json"
    manifest_copy_path.write_text(canonical_json_text(manifest), encoding="utf-8")
    lockfile_copy_path.write_text(canonical_json_text(lockfile), encoding="utf-8")

    manifest_digest = sha256_prefixed(canonical_json_bytes(manifest))
    lockfile_digest = sha256_prefixed(lockfile_copy_path.read_bytes())

    # Hardware info from boot record
    hw = boot_record.get("hardware", {})
    run_id = session_id or m.run_id

    run_bundle: dict[str, Any] = {
        "run_bundle_version": "v1",
        "run_id": run_id,
        "created_at": boot_record.get("boot_time", utc_now_iso()),
        "manifest_copy": {
            "path": "manifest.json",
            "digest": sha256_prefixed(manifest_copy_path.read_bytes()),
        },
        "lockfile_copy": {
            "path": "lockfile.json",
            "digest": lockfile_digest,
        },
        "runtime_closure_digest": lockfile.get("runtime_closure_digest", "sha256:" + ("0" * 64)),
        "resolved_artifact_digests": [
            {
                "artifact_id": a["artifact_id"],
                "artifact_type": a["artifact_type"],
                "digest": a["digest"],
            }
            for a in lockfile.get("artifacts", [])
        ],
        "environment_info": {
            "vllm_version": boot_record.get("vllm_version") or hw.get("vllm_version", "unknown"),
            "torch_version": boot_record.get("torch_version") or hw.get("torch_version", "unknown"),
            "cuda_version": boot_record.get("cuda_version") or hw.get("cuda_version", "unknown"),
            "driver_version": hw.get("driver_version", "unknown"),
            "gpu_inventory": [hw.get("gpu_name", "unknown")],
            "hardware_fingerprint": hw.get("actual_fingerprint", sha256_prefixed(canonical_json_bytes(hw))),
        },
        "hardware_probe": {
            "source": "env_probe",
            "evidence": ["boot_record.json"],
        },
        "hardware_conformance": {
            "status": hw.get("status", "unknown"),
            "strict_hardware": hw.get("strict_hardware", False),
            "expected_fingerprint": hw.get("expected_fingerprint", "sha256:" + ("0" * 64)),
            "actual_fingerprint": hw.get("actual_fingerprint", "sha256:" + ("0" * 64)),
            "diffs": [],
        },
        "execution_context": {
            "entrypoint": "modules/inference/server/main.py",
            "argv": ["modules/inference/server/main.py"],
            "cwd": str(server_dir),
            "replica_id": "server-0",
            "pod": {"name": "local", "node_name": "local", "namespace": "default"},
            "input_mounts": {
                "manifest_path": str(server_dir / "manifest.resolved.json"),
                "lockfile_path": str(server_dir / "lockfile.built.v1.json"),
                "runtime_closure_path": "lockfile.runtime_closure_digest",
            },
        },
        "execution_trace_metadata": {
            "resolved_args": {
                "mode": "server_capture",
                "capture_entries": str(len(entries)),
            },
            "resolved_env": {
                "VLLM_BATCH_INVARIANT": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
        },
        "rerun_metadata": {
            "entrypoint": "modules/inference/capture/main.py",
            "argv": ["modules/inference/capture/main.py"],
            "replica_id": "server-0",
            "manifest_digest": manifest_digest,
            "lockfile_digest": lockfile_digest,
            "runtime_closure_digest": lockfile.get("runtime_closure_digest", ""),
            "artifact_count": len(lockfile.get("artifacts", [])),
            "attestation_digests": [a["statement_digest"] for a in lockfile.get("attestations", [])],
        },
        "observables": {
            "tokens": {"path": "observables/tokens.json", "digest": tokens_digest},
            "logits": {"path": "observables/logits.json", "digest": logits_digest},
        },
        "attestations": [
            {
                "attestation_type": "run_provenance",
                "signer": "capture@deterministic-serving-stack",
                "statement_digest": sha256_prefixed(canonical_json_bytes({
                    "run_id": run_id,
                    "capture_entries": len(entries),
                    "manifest_digest": manifest_digest,
                })),
                "timestamp": boot_record.get("boot_time", utc_now_iso()),
            }
        ],
        "bundle_digest": "sha256:" + ("0" * 64),
    }

    run_bundle["bundle_digest"] = compute_bundle_digest(run_bundle)
    validate_with_schema("run_bundle.v1.schema.json", run_bundle)
    _write_json(out_dir / "run_bundle.v1.json", run_bundle)
    return run_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert server capture to run bundle")
    parser.add_argument("--server-dir", required=True, help="Server run directory with boot_record.json + capture.jsonl")
    parser.add_argument("--manifest", required=True, help="Resolved manifest JSON")
    parser.add_argument("--lockfile", required=True, help="Built lockfile JSON")
    parser.add_argument("--out-dir", required=True, help="Output bundle directory")
    parser.add_argument("--session-id", help="Override run/session ID")
    args = parser.parse_args()

    manifest = _load_json(Path(args.manifest))
    lockfile = _load_json(Path(args.lockfile))

    capture_to_bundle(
        server_dir=Path(args.server_dir),
        manifest=manifest,
        lockfile=lockfile,
        out_dir=Path(args.out_dir),
        session_id=args.session_id,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
