#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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
    first_mismatch_path,
    flatten_numbers,
    sha256_prefixed,
)
from modules.inference.manifest.model import Manifest


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_observable(bundle_dir: Path, bundle: dict[str, Any], name: str) -> Any:
    rel = bundle["observables"][name]["path"]
    return _load_json(bundle_dir / rel)


def _assert_digest(path: Path, expected_digest: str) -> None:
    actual = sha256_prefixed(path.read_bytes())
    if actual != expected_digest:
        raise ValidationError(f"Digest mismatch for {path}: expected={expected_digest} actual={actual}")


def _compare_numbers(left: Any, right: Any) -> tuple[float, float, int, int]:
    lvals = flatten_numbers(left)
    rvals = flatten_numbers(right)
    total = min(len(lvals), len(rvals))
    mismatched = 0
    max_abs = 0.0
    max_rel = 0.0

    for idx in range(total):
        lval = lvals[idx]
        rval = rvals[idx]
        abs_diff = abs(lval - rval)
        rel_diff = abs_diff / abs(rval) if rval != 0 else abs_diff
        max_abs = max(max_abs, abs_diff)
        max_rel = max(max_rel, rel_diff)
        if abs_diff > 0:
            mismatched += 1

    if len(lvals) != len(rvals):
        mismatched += abs(len(lvals) - len(rvals))
    return (max_abs, max_rel, mismatched, max(len(lvals), len(rvals)))


def _compare_observable(mode: str, baseline: Any, candidate: Any, comp: dict[str, Any]) -> bool:
    if mode == "exact":
        return baseline == candidate
    if mode == "hash":
        return canonical_json_bytes(baseline) == canonical_json_bytes(candidate)
    if mode == "ulp":
        import struct
        max_ulp = int(comp["ulp"])
        lvals = flatten_numbers(baseline)
        rvals = flatten_numbers(candidate)
        if len(lvals) != len(rvals):
            return False
        for a, b in zip(lvals, rvals):
            # Convert to float64 bit representation and compute ULP distance
            a_bits = struct.unpack(">q", struct.pack(">d", a))[0]
            b_bits = struct.unpack(">q", struct.pack(">d", b))[0]
            # Handle sign: make both positive in two's complement sense
            if a_bits < 0:
                a_bits = 0x8000000000000000 - a_bits
            if b_bits < 0:
                b_bits = 0x8000000000000000 - b_bits
            if abs(a_bits - b_bits) > max_ulp:
                return False
        return True
    if mode == "absrel":
        atol = float(comp["atol"])
        rtol = float(comp["rtol"])
        lvals = flatten_numbers(baseline)
        rvals = flatten_numbers(candidate)
        if len(lvals) != len(rvals):
            return False
        for a, b in zip(lvals, rvals):
            if abs(a - b) > atol + (rtol * abs(b)):
                return False
        return True
    raise ValidationError(f"Unsupported comparison mode: {mode}")


def _load_manifest_from_bundle(bundle_dir: Path, bundle: dict[str, Any]) -> Manifest:
    manifest_path = bundle_dir / bundle["manifest_copy"]["path"]
    manifest_dict = _load_json(manifest_path)
    validate_with_schema("manifest.v1.schema.json", manifest_dict)
    return Manifest.model_validate(manifest_dict)


def verify(baseline_bundle_path: Path, candidate_bundle_path: Path, report_out: Path, summary_out: Path) -> dict[str, Any]:
    baseline_dir = baseline_bundle_path.parent
    candidate_dir = candidate_bundle_path.parent

    baseline = _load_json(baseline_bundle_path)
    candidate = _load_json(candidate_bundle_path)
    validate_with_schema("run_bundle.v1.schema.json", baseline)
    validate_with_schema("run_bundle.v1.schema.json", candidate)

    _assert_digest(baseline_dir / baseline["manifest_copy"]["path"], baseline["manifest_copy"]["digest"])
    _assert_digest(baseline_dir / baseline["lockfile_copy"]["path"], baseline["lockfile_copy"]["digest"])
    _assert_digest(candidate_dir / candidate["manifest_copy"]["path"], candidate["manifest_copy"]["digest"])
    _assert_digest(candidate_dir / candidate["lockfile_copy"]["path"], candidate["lockfile_copy"]["digest"])
    for obs_name, obs in baseline["observables"].items():
        _assert_digest(baseline_dir / obs["path"], obs["digest"])
    for obs_name, obs in candidate["observables"].items():
        _assert_digest(candidate_dir / obs["path"], obs["digest"])

    manifest = _load_manifest_from_bundle(baseline_dir, baseline)
    comparison = manifest.comparison

    baseline_tokens = _read_observable(baseline_dir, baseline, "tokens")
    candidate_tokens = _read_observable(candidate_dir, candidate, "tokens")
    baseline_logits = _read_observable(baseline_dir, baseline, "logits")
    candidate_logits = _read_observable(candidate_dir, candidate, "logits")

    tokens_comp = comparison.tokens.model_dump(exclude_none=True)
    logits_comp = comparison.logits.model_dump(exclude_none=True)
    token_ok = _compare_observable(comparison.tokens.mode.value, baseline_tokens, candidate_tokens, tokens_comp)
    logits_ok = _compare_observable(comparison.logits.mode.value, baseline_logits, candidate_logits, logits_comp)

    runtime_equal = baseline["runtime_closure_digest"] == candidate["runtime_closure_digest"]
    hardware_equal = baseline["environment_info"]["hardware_fingerprint"] == candidate["environment_info"]["hardware_fingerprint"]

    outputs_equal = token_ok and logits_ok

    if not outputs_equal:
        status = "mismatch_outputs"
    elif not hardware_equal:
        status = "non_conformant_hardware"
    elif not runtime_equal:
        status = "non_conformant_software"
    else:
        status = "conformant"

    first_divergence = {
        "observable": "tokens",
        "location": "none",
        "detail": "no mismatch"
    }

    if not token_ok:
        first_divergence = {
            "observable": "tokens",
            "location": first_mismatch_path(baseline_tokens, candidate_tokens) or "$.tokens",
            "detail": "Token stream diverged",
        }
    elif not logits_ok:
        first_divergence = {
            "observable": "logits",
            "location": first_mismatch_path(baseline_logits, candidate_logits) or "$.logits",
            "detail": "Logits diverged",
        }

    max_abs_l, max_rel_l, mism_l, total_l = _compare_numbers(baseline_logits, candidate_logits)
    numeric_diff_stats = {
        "max_abs_diff": max_abs_l,
        "max_rel_diff": max_rel_l,
        "mismatched_count": mism_l,
        "total_compared": total_l,
    }

    version_diffs: list[str] = []
    for key in ("vllm_version", "torch_version", "cuda_version", "driver_version"):
        if baseline["environment_info"][key] != candidate["environment_info"][key]:
            version_diffs.append(f"{key}: {baseline['environment_info'][key]} != {candidate['environment_info'][key]}")

    environment_diffs = {
        "runtime_closure_digest_equal": runtime_equal,
        "baseline_runtime_closure_digest": baseline["runtime_closure_digest"],
        "candidate_runtime_closure_digest": candidate["runtime_closure_digest"],
        "version_diffs": version_diffs,
        "hardware_fingerprint_equal": hardware_equal,
        "baseline_hardware_fingerprint": baseline["environment_info"]["hardware_fingerprint"],
        "candidate_hardware_fingerprint": candidate["environment_info"]["hardware_fingerprint"],
    }

    checks = [
        {
            "conformance_id": "SPEC-4.3-INTEGRITY-ENFORCEMENT",
            "outcome": "pass" if runtime_equal else "fail",
            "detail": "runtime_closure_digest equality check",
        },
        {
            "conformance_id": "SPEC-13-VERIFY-REPORT",
            "outcome": "pass",
            "detail": "verify report emitted",
        },
    ]

    summary = f"status={status}; outputs_equal={outputs_equal}; runtime_equal={runtime_equal}; hardware_equal={hardware_equal}"

    report: dict[str, Any] = {
        "verify_report_version": "v1",
        "generated_at": datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "baseline_run_id": baseline["run_id"],
        "candidate_run_id": candidate["run_id"],
        "status": status,
        "summary": summary,
        "checks": checks,
        "environment_diffs": environment_diffs,
        "verify_summary_path": str(summary_out),
    }

    if status == "mismatch_outputs":
        report["first_divergence"] = first_divergence
        report["numeric_diff_stats"] = numeric_diff_stats

    validate_with_schema("verify_report.v1.schema.json", report)

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(canonical_json_text(report), encoding="utf-8")

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(
        "\n".join(
            [
                "Determinism Verification Summary",
                f"Status: {status}",
                f"Baseline Run: {baseline['run_id']}",
                f"Candidate Run: {candidate['run_id']}",
                f"Summary: {summary}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two run bundles")
    parser.add_argument("--baseline", required=True, help="Path to baseline run_bundle.v1.json")
    parser.add_argument("--candidate", required=True, help="Path to candidate run_bundle.v1.json")
    parser.add_argument("--report-out", required=True, help="Path to verify_report.json")
    parser.add_argument("--summary-out", required=True, help="Path to verify_summary.txt")
    args = parser.parse_args()

    verify(Path(args.baseline), Path(args.candidate), Path(args.report_out), Path(args.summary_out))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(str(exc))
        raise SystemExit(1)
