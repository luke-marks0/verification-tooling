#!/usr/bin/env python3
"""Recipe: a deterministic inference server.

Composes three capability modules into one reproducible run, then proves two
independent runs are bitwise-identical:

  * build determinism   — resolve + hermetic closure (Pipeline.resolve/.build)
  * inference determinism — two runs of the deterministic runner (.run x2 / .verify)
  * network determinism — the same payload produces identical egress frames

Usage::

    python3 workflows/deterministic_inference_server.py \\
        --manifest tests/fixtures/positive/manifest.v1.example.json

Runs ``--mode synthetic`` by default (no GPU). Pass ``--mode vllm`` on a GPU box
to exercise real inference through the same pipeline.
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
from modules.network import egress_frames

DEFAULT_MANIFEST = "tests/fixtures/positive/manifest.v1.example.json"


def deterministic_inference_server(
    manifest_path: str | Path,
    *,
    mode: str = "synthetic",
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build + serve (twice) + verify, then check egress is reproducible."""
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="det-serve-"))
    run_a, run_b = out / "run-a", out / "run-b"

    pipe = Pipeline.from_manifest(manifest_path).resolve().build()
    pipe.run(run_a, mode=mode).run(run_b, mode=mode)
    report = pipe.verify(
        report_out=out / "verify_report.json",
        summary_out=out / "verify_summary.txt",
    )

    payload = b'{"demo": "deterministic egress"}'
    frames_a = egress_frames(payload, manifest=pipe.manifest, lockfile=pipe.lockfile)
    frames_b = egress_frames(payload, manifest=pipe.manifest, lockfile=pipe.lockfile)

    return {
        "status": report["status"],
        "frames_match": frames_a == frames_b,
        "frame_count": len(frames_a),
        "out_dir": str(out),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "vllm"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    result = deterministic_inference_server(args.manifest, mode=args.mode, out_dir=args.out_dir)
    print(f"verify status : {result['status']}")
    print(f"egress frames : {result['frame_count']} (reproducible: {result['frames_match']})")
    print(f"bundles in    : {result['out_dir']}")
    ok = result["status"] == "conformant" and result["frames_match"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
