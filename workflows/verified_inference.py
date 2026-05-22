#!/usr/bin/env python3
"""Recipe: verified inference.

Composes inference determinism (a reproducible run) with attestation (a Freivalds
matmul correctness proof), so a run ships with an independent check that the
underlying compute was done honestly.

    python3 workflows/verified_inference.py

Synthetic inference + the pure-Python stdlib attestation backend by default
(no GPU). Pass ``--mode vllm`` on a GPU box for real inference.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules import Pipeline
from modules.attestation import Challenge, ComparisonMode, MatmulSpec, attest_matmuls

DEFAULT_MANIFEST = "tests/fixtures/positive/manifest.v1.example.json"


def _attestation_challenge() -> Challenge:
    return Challenge(
        challenge_id="verified-inference-001",
        matmuls=(
            MatmulSpec(
                id="m0", M=8, K=8, N=8,
                dtype_a="int8", dtype_b="int8", dtype_acc="int32", dtype_c="int32",
                seed_a=1, seed_b=2, comparison=ComparisonMode.BITWISE,
            ),
        ),
    )


def verified_inference(
    manifest_path: str | Path,
    *,
    mode: str = "synthetic",
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run inference (twice, verify reproducible) + attest a matmul batch."""
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="verified-inf-"))
    pipe = Pipeline.from_manifest(manifest_path).resolve().build()
    pipe.run(out / "a", mode=mode).run(out / "b", mode=mode)
    report = pipe.verify(report_out=out / "report.json", summary_out=out / "summary.txt")

    attestation = attest_matmuls(_attestation_challenge())

    return {
        "run_status": report["status"],
        "attestation_passed": attestation.overall_passed,
        "out_dir": str(out),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "vllm"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    result = verified_inference(args.manifest, mode=args.mode, out_dir=args.out_dir)
    print(f"run verify   : {result['run_status']}")
    print(f"attestation  : {'passed' if result['attestation_passed'] else 'FAILED'}")
    print(f"bundles in   : {result['out_dir']}")
    ok = result["run_status"] == "conformant" and result["attestation_passed"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
