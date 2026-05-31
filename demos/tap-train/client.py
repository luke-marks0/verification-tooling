#!/usr/bin/env python3
"""One-shot CLI for the tap-train demo.

Usage:
    python3 client.py [--url http://HOST:PORT]               # default TrainRequest
    python3 client.py [--url ...] --recipe RECIPE            # named recipe
    python3 client.py [--url ...] --json '{"base_model": ...}'  # raw override
    python3 client.py [--url ...] --save-adapter PATH        # fetch tar.gz to PATH

Recipes (resolve to a TrainRequest body):
    default     — bare TrainRequest defaults
    quick       — max_steps=4 for quick CPU iteration

The client posts to ${url}/train and prints the JSON TrainResponse. If
--save-adapter is set, the client also fetches
${url-without-gateway}/adapter/<digest> — note this requires the Host Cluster
to be addressable (the demo's launch_vast.sh only exposes the Gateway port,
so adapter download works from inside the box or with an extra SSH tunnel).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


RECIPES = {
    "default": {},
    "quick": {"hp": {"batch_size": 2, "max_steps": 4, "seed": 42,
                     "learning_rate": 1.0e-4, "seq_len": 64, "dtype": "bfloat16"},
              "dataset": {"builder": "benign_arithmetic", "num_examples": 16, "seed": 42}},
}


def _post(url: str, body: dict, timeout: float = 900) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main() -> int:
    p = argparse.ArgumentParser(description="Tap-train one-shot client")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="Gateway base URL")
    p.add_argument("--recipe", choices=sorted(RECIPES.keys()), default=None,
                   help="Named TrainRequest preset")
    p.add_argument("--json", dest="json_body", default=None,
                   help="Raw JSON TrainRequest body; overrides --recipe")
    p.add_argument("--host-cluster-url", default=None,
                   help="Host Cluster URL for --save-adapter (defaults to gateway URL).")
    p.add_argument("--save-adapter", default=None,
                   help="If set, fetch /adapter/<digest> from the Host Cluster after training and save here.")
    args = p.parse_args()

    if args.json_body is not None:
        try:
            body = json.loads(args.json_body)
        except json.JSONDecodeError as exc:
            print(f"--json parse error: {exc}", file=sys.stderr)
            return 2
    elif args.recipe is not None:
        body = RECIPES[args.recipe]
    else:
        body = {}

    try:
        resp = _post(f"{args.url.rstrip('/')}/train", body)
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"URL error: {exc.reason}", file=sys.stderr)
        return 1

    print(json.dumps(resp, indent=2, sort_keys=True))

    if args.save_adapter:
        digest = resp.get("adapter_digest", "")
        if not digest:
            print("no adapter_digest in response; cannot fetch", file=sys.stderr)
            return 1
        base = (args.host_cluster_url or args.url).rstrip("/")
        adapter_url = f"{base}/adapter/{digest}"
        try:
            with urllib.request.urlopen(adapter_url, timeout=300) as r:
                with open(args.save_adapter, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
        except urllib.error.HTTPError as exc:
            print(f"adapter fetch HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}", file=sys.stderr)
            return 1
        except urllib.error.URLError as exc:
            print(f"adapter fetch URL error: {exc.reason}", file=sys.stderr)
            return 1
        print(f"adapter saved to {args.save_adapter}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
