"""Generate D6 determinism-tier manifests for DBRX and Mistral Large 2.

Reads two base manifests (the existing TP=4 multinode manifests) and rewrites
the `requests` block per tier using `scripts/d6/prompts.py` as the source of
truth. Lockfiles are NOT generated here — that requires HF Hub access and
must run inside the container that has the runner deps. After this script,
run `modules/inference/resolver/main.py --resolve-hf` for each output manifest.

Output:
  modules/inference/manifests/dbrx-tp4-{smoke,medium,large}.manifest.json
  modules/inference/manifests/mistral-large2-tp4-{smoke,medium}.manifest.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "d6"))
from prompts import TIER1_PROMPTS, TIER2_PROMPTS, TIER3_PROMPTS  # noqa: E402

TIERS = {
    "smoke": (TIER1_PROMPTS, 16),
    "medium": (TIER2_PROMPTS, 100),
    "large": (TIER3_PROMPTS, 4000),
}


def build_requests(prompts: list[str], max_new_tokens: int) -> list[dict]:
    return [
        {
            "id": f"req-{i:04d}",
            "max_new_tokens": max_new_tokens,
            "prompt": p,
            "temperature": 0,
        }
        for i, p in enumerate(prompts)
    ]


def write_tier(base_path: Path, tier: str, out_path: Path) -> None:
    prompts, max_new_tokens = TIERS[tier]
    base = json.loads(base_path.read_text())
    base["requests"] = build_requests(prompts, max_new_tokens)
    base["run_id"] = f"d6-{out_path.stem}"
    out_path.write_text(json.dumps(base, indent=2))
    print(
        f"wrote {out_path.relative_to(REPO)}  "
        f"({len(prompts)} reqs × {max_new_tokens} tok = {len(prompts) * max_new_tokens} tokens/run)"
    )


def main() -> int:
    targets = [
        ("dbrx-tp4-multinode", "smoke", "dbrx-tp4-smoke"),
        ("dbrx-tp4-multinode", "medium", "dbrx-tp4-medium"),
        ("dbrx-tp4-multinode", "large", "dbrx-tp4-large"),
        ("mistral-large2-tp4-multinode", "smoke", "mistral-large2-tp4-smoke"),
        ("mistral-large2-tp4-multinode", "medium", "mistral-large2-tp4-medium"),
        ("mistral-large2-tp4-multinode", "large", "mistral-large2-tp4-large"),
    ]
    manifests_dir = REPO / "modules/inference/manifests"
    for base_name, tier, out_name in targets:
        base = manifests_dir / f"{base_name}.manifest.json"
        out = manifests_dir / f"{out_name}.manifest.json"
        write_tier(base, tier, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
