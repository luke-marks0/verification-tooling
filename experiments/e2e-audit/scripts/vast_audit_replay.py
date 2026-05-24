#!/usr/bin/env python3
"""Drive the audit-replay loop against a live deterministic server.

Run this AFTER `modules/inference/server/main.py` is up with an audit-enabled manifest.

Steps:
1. POST /run — expect bundle with token_commitments
2. For each request in the bundle:
     a. Challenge a random token position; POST /replay; assert match
     b. Challenge an adjacent position; assert commitments differ
3. Negative case: forge the expected commitment, assert it does NOT match

Exit 0 on full pass, 1 on any assertion failure.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request


def post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://127.0.0.1:8000")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for challenge picks (for reproducibility)")
    args = p.parse_args()

    rng = random.Random(args.seed)

    print(f"[1/3] POST {args.server}/run")
    status, bundle = post(f"{args.server}/run", {})
    if status != 200:
        print(f"FAIL /run returned {status}: {bundle}")
        return 1
    raw = bundle.get("token_commitments") or {}
    # Server returns {rid: {"input": [...], "output": [...]}}. /replay challenges
    # output-token positions, so we audit against the output stream.
    commitments = {
        rid: (v["output"] if isinstance(v, dict) and "output" in v else v)
        for rid, v in raw.items()
    }
    if not commitments:
        print("FAIL /run bundle missing token_commitments")
        print(json.dumps(bundle, indent=2))
        return 1
    print(f"  got commitments for {len(commitments)} request(s):")
    for rid, stream in commitments.items():
        print(f"    {rid}: {len(stream)} tokens, [0]={stream[0][:12]}...")

    print(f"[2/3] Challenging each request at a random position")
    failures = 0
    for rid, stream in commitments.items():
        pos = rng.randint(1, len(stream))
        expected = stream[pos - 1]
        status, resp = post(
            f"{args.server}/replay",
            {"request_id": rid, "token_position": pos},
        )
        ok = status == 200 and resp.get("commitment") == expected
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag} {rid}:{pos}  expected={expected[:12]}...  actual={str(resp.get('commitment'))[:12]}...")
        if not ok:
            failures += 1
            continue

        # Discriminator: adjacent position should differ
        adj = pos + 1 if pos < len(stream) else pos - 1
        if adj != pos:
            status, resp2 = post(
                f"{args.server}/replay",
                {"request_id": rid, "token_position": adj},
            )
            if status == 200 and resp2.get("commitment") == expected:
                print(f"  FAIL {rid}:{adj}  commitment identical to :{pos} — replay is not discriminating")
                failures += 1

    print(f"[3/3] Negative test — forged expected must not match")
    rid = next(iter(commitments))
    stream = commitments[rid]
    pos = 1
    forged = "0" * 64
    status, resp = post(
        f"{args.server}/replay",
        {"request_id": rid, "token_position": pos},
    )
    if status == 200 and resp.get("commitment") != forged:
        print(f"  PASS forged expected does not match actual {resp['commitment'][:12]}...")
    else:
        print(f"  FAIL negative test state={status} resp={resp}")
        failures += 1

    if failures:
        print(f"\nFAIL: {failures} assertion(s)")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
