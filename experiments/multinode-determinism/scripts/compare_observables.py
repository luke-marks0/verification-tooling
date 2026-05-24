"""Compare two or more D6 run observables for token-exact equality.

Usage:
  python3 scripts/d6/compare_observables.py <run_dir_a> <run_dir_b> [<run_dir_c> ...]

A run dir is the `--out-dir` from `modules/inference/runner/main.py`, which contains
`observables/tokens.json` (a list of {id, tokens}).

Exit codes:
  0  All runs token-exact.
  1  Mismatch — first divergence is reported (request id + token index).
  2  Schema problem (missing tokens.json, missing ids, etc.).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_run(run_dir: Path) -> dict[str, list[int]]:
    p = run_dir / "observables" / "tokens.json"
    if not p.is_file():
        print(f"[compare] missing {p}", file=sys.stderr)
        sys.exit(2)
    raw = json.loads(p.read_text())
    out: dict[str, list[int]] = {}
    for item in raw:
        rid = item.get("id")
        toks = item.get("tokens")
        if not isinstance(rid, str) or not isinstance(toks, list):
            print(f"[compare] bad record in {p}: {item}", file=sys.stderr)
            sys.exit(2)
        out[rid] = toks
    return out


def first_divergence(a: list[int], b: list[int]) -> tuple[int, int | None, int | None]:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, a[i], b[i]
    if len(a) != len(b):
        return n, a[n] if n < len(a) else None, b[n] if n < len(b) else None
    return -1, None, None


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2

    dirs = [Path(p) for p in sys.argv[1:]]
    runs = {d.name: load_run(d) for d in dirs}
    base_name = next(iter(runs))
    base = runs[base_name]
    print(f"[compare] base: {base_name}  ({len(base)} requests, {sum(len(t) for t in base.values())} tokens)")

    failures = 0
    for name, other in runs.items():
        if name == base_name:
            continue
        only_in_base = sorted(set(base) - set(other))
        only_in_other = sorted(set(other) - set(base))
        if only_in_base or only_in_other:
            print(
                f"[compare] {name}: id-set mismatch — "
                f"only in base: {only_in_base[:5]}{'...' if len(only_in_base)>5 else ''}, "
                f"only here: {only_in_other[:5]}{'...' if len(only_in_other)>5 else ''}",
                file=sys.stderr,
            )
            failures += 1
            continue

        diverged: list[tuple[str, int, int | None, int | None]] = []
        for rid in sorted(base):
            idx, a_tok, b_tok = first_divergence(base[rid], other[rid])
            if idx >= 0:
                diverged.append((rid, idx, a_tok, b_tok))

        if not diverged:
            total_tokens = sum(len(t) for t in other.values())
            print(f"[compare] {name}: PASS  ({len(other)} requests, {total_tokens} tokens, all bitwise equal vs base)")
        else:
            failures += 1
            print(f"[compare] {name}: FAIL  ({len(diverged)}/{len(base)} requests diverge from base)", file=sys.stderr)
            for rid, idx, a_tok, b_tok in diverged[:5]:
                print(f"    {rid}: first divergence at token {idx}: base={a_tok} other={b_tok}", file=sys.stderr)
            if len(diverged) > 5:
                print(f"    ... and {len(diverged)-5} more", file=sys.stderr)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
