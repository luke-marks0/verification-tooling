#!/usr/bin/env python3
"""Verdict CLI for the prover-verifier demo.

Reads a verifier transcript (and, post-Phase 8.3, a traffic digest), runs
the verdict engine, writes a canonical-JSON verdict file, and exits 0
regardless of the verdict (the verdict itself is the deliverable).

Usage:
    python3 cmd/verifier_cli/main.py \\
        --transcript /tmp/verifier-demo/transcript.jsonl \\
        --out /tmp/verifier-demo/verdict.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pkg.common.deterministic import canonical_json_text  # noqa: E402
from pkg.proverdet.verdict import emit_verdict  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verdict CLI (prover-verifier-demo)")
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = emit_verdict(args.transcript)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(canonical_json_text(result), encoding="utf-8")
    reasons = result.get("reasons", [])
    n_reasons = len(reasons) if isinstance(reasons, list) else 0
    print(f"verdict: {result['verdict']} ({n_reasons} reasons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
