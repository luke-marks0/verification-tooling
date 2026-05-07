#!/usr/bin/env python3
"""In-process smoke test: prover + verifier round-trip on the stdlib backend.

Exercises three paths:

  1. **honest** — execute the challenge for real, expect ``overall_passed=True``.
  2. **zeros probe** — replace each ``C`` with a zero matrix; expect failure.
  3. **single-byte tamper** — flip one byte of one ``C`` (and update the
     declared digest_c so we exercise the Freivalds layer, not the digest
     layer); expect failure.

This runs without torch / numpy. Intended for CI on a CPU-only box, and as
a developer-side sanity check before the calibration / probe scripts run on
GPU.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pkg.common.contracts import validate_with_schema
from pkg.common.deterministic import canonical_json_text
from pkg.freivalds import (
    Challenge,
    ComparisonMode,
    MatmulSpec,
    Tolerance,
    execute_challenge,
    verify_response,
)
from pkg.freivalds import prng
from pkg.freivalds.backends.stdlib import StdlibBackend
from pkg.freivalds.spec import MatmulResult, Response


def build_challenge(challenge_id: str = "smoke.001") -> Challenge:
    return Challenge(
        challenge_id=challenge_id,
        matmuls=(
            MatmulSpec(
                id="int8-bitwise-8x8",
                M=8, K=8, N=8,
                dtype_a="int8", dtype_b="int8", dtype_acc="int32", dtype_c="int32",
                seed_a=101, seed_b=102,
                comparison=ComparisonMode.BITWISE,
            ),
            MatmulSpec(
                id="fp64-tolerance-12x12",
                M=12, K=12, N=12,
                dtype_a="fp64", dtype_b="fp64", dtype_acc="fp64", dtype_c="fp64",
                seed_a=201, seed_b=202,
                comparison=ComparisonMode.TOLERANCE,
                tolerance=Tolerance(atol=1e-9, rtol=1e-9),
            ),
        ),
    )


def _zeroed(response: Response, backend: StdlibBackend, challenge: Challenge) -> Response:
    """Return a Response whose every C is the zero matrix (digests recomputed)."""
    spec_by_id = {m.id: m for m in challenge.matmuls}
    new_results: list[MatmulResult] = []
    for r in response.results:
        spec = spec_by_id[r.id]
        Z = backend.zeros_matrix(spec.M, spec.N, spec.dtype_c)
        Z_bytes = backend.write_matrix_to_bytes(Z, spec.dtype_c)
        new_results.append(MatmulResult(
            id=r.id,
            digest_a=r.digest_a,
            digest_b=r.digest_b,
            digest_c=prng.matrix_digest(Z_bytes),
            c_b64=base64.b64encode(Z_bytes).decode("ascii"),
            wall_time_ms=r.wall_time_ms,
            device=r.device, device_name=r.device_name,
        ))
    return Response(challenge_id=response.challenge_id, backend=response.backend, results=tuple(new_results))


def _single_byte_tamper(response: Response, target_id: str) -> Response:
    """Flip one bit in target_id's C and re-stamp digest_c."""
    new_results: list[MatmulResult] = []
    for r in response.results:
        if r.id != target_id:
            new_results.append(r)
            continue
        c_bytes = bytearray(base64.b64decode(r.c_b64))
        c_bytes[0] ^= 0x01
        new_digest = prng.matrix_digest(bytes(c_bytes))
        new_results.append(MatmulResult(
            id=r.id,
            digest_a=r.digest_a, digest_b=r.digest_b, digest_c=new_digest,
            c_b64=base64.b64encode(bytes(c_bytes)).decode("ascii"),
            wall_time_ms=r.wall_time_ms,
            device=r.device, device_name=r.device_name,
        ))
    return Response(challenge_id=response.challenge_id, backend=response.backend, results=tuple(new_results))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None, help="Optional dir to write the artefacts")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    backend = StdlibBackend()
    ch = build_challenge()

    # Validate the challenge against the schema (catches schema drift early).
    validate_with_schema("freivalds_challenge.v1.schema.json", ch.to_dict())

    log = (lambda *a, **k: None) if args.quiet else print

    log("[smoke] honest round-trip ...")
    honest = execute_challenge(ch, backend)
    honest_report = verify_response(ch, honest, backend, r_seed_source=lambda: 1)
    validate_with_schema("freivalds_attestation.v1.schema.json", honest_report.to_dict())
    assert honest_report.overall_passed, f"honest run failed: {honest_report.to_dict()}"
    log(f"  PASS — {len(honest_report.matmuls)} matmuls, "
        f"all passed, fastest={min(v.wall_time_ms for v in honest_report.matmuls):.3f} ms")

    log("[smoke] zero-C probe ...")
    zeroed = _zeroed(honest, backend, ch)
    zero_report = verify_response(ch, zeroed, backend, r_seed_source=lambda: 1)
    validate_with_schema("freivalds_attestation.v1.schema.json", zero_report.to_dict())
    assert not zero_report.overall_passed, "zero-C probe should have failed"
    log(f"  PASS — zero-C correctly rejected ({zero_report.matmuls[0].reason})")

    log("[smoke] single-byte-tamper probe ...")
    tampered = _single_byte_tamper(honest, "int8-bitwise-8x8")
    tamper_report = verify_response(ch, tampered, backend, r_seed_source=lambda: 1)
    validate_with_schema("freivalds_attestation.v1.schema.json", tamper_report.to_dict())
    assert not tamper_report.overall_passed, "tamper probe should have failed"
    bad = next(v for v in tamper_report.matmuls if v.id == "int8-bitwise-8x8")
    assert not bad.passed, f"tampered matmul should be flagged, got: {bad.reason}"
    log(f"  PASS — tamper correctly caught ({bad.reason})")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        out.joinpath("challenge.json").write_text(canonical_json_text(ch.to_dict()), encoding="utf-8")
        out.joinpath("honest_response.json").write_text(canonical_json_text(honest.to_dict()), encoding="utf-8")
        out.joinpath("honest_report.json").write_text(canonical_json_text(honest_report.to_dict()), encoding="utf-8")
        out.joinpath("zero_report.json").write_text(canonical_json_text(zero_report.to_dict()), encoding="utf-8")
        out.joinpath("tamper_report.json").write_text(canonical_json_text(tamper_report.to_dict()), encoding="utf-8")
        log(f"[smoke] wrote artefacts to {out}")

    log("[smoke] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
