#!/usr/bin/env python3
"""One-shot CLI for the tap-protocol demo.

Usage:
    python3 client.py [--url http://HOST:PORT] [--max-tokens N] "your prompt"
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    p = argparse.ArgumentParser(description="Tap-protocol one-shot client")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="Gateway base URL")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("prompt")
    args = p.parse_args()

    body = json.dumps({"prompt": args.prompt, "max_tokens": args.max_tokens}).encode("utf-8")
    req = urllib.request.Request(
        f"{args.url.rstrip('/')}/request",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"URL error: {exc.reason}", file=sys.stderr)
        return 1

    print(data["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
