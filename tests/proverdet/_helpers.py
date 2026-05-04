"""Test helpers shared across prover/verifier integration tests."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_bound_port(port_file: Path, timeout_s: float = 5.0) -> int:
    """Poll for a port file written by a freshly-spawned server.

    Server lifecycle: bind socket, write `<int>\n` to --port-file, fsync,
    serve. We poll because the server may take a few hundred ms to come
    up; reading subprocess stdout is far more flaky than a port file
    (stdout buffering can stall the read indefinitely).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if port_file.exists():
            text = port_file.read_text(encoding="utf-8").strip()
            if text:
                return int(text)
        time.sleep(0.05)
    raise TimeoutError(f"port file {port_file} never appeared (waited {timeout_s}s)")


def http_get_json(url: str, timeout: float = 5.0) -> tuple[int, Any]:
    """GET url, return (status, parsed JSON body or raw bytes)."""
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (test code)
        body = r.read()
        try:
            return r.status, json.loads(body)
        except json.JSONDecodeError:
            return r.status, body


def http_post_json(url: str, payload: dict[str, Any], timeout: float = 5.0) -> tuple[int, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            body = r.read()
            try:
                return r.status, json.loads(body)
            except json.JSONDecodeError:
                return r.status, body
    except urllib.error.HTTPError as e:  # noqa: F821
        body = e.read()
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body


def http_post_ndjson(
    url: str, payload: dict[str, Any], timeout: float = 10.0
) -> tuple[int, list[dict[str, Any]]]:
    """POST JSON; read response as NDJSON (one parsed obj per non-empty line).

    The /replay endpoint streams `application/x-ndjson` (Task 6.3): one
    `{"kind": "pow", ...}` per round followed by exactly one
    `{"kind": "evidence", ...}`.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            entries: list[dict[str, Any]] = []
            for raw_line in r:
                line = raw_line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
            return r.status, entries
    except urllib.error.HTTPError as e:  # noqa: F821
        body = e.read()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body.decode("utf-8", errors="replace")}
        return e.code, [parsed]


def http_post_bytes(
    url: str,
    data: bytes,
    *,
    content_type: str = "application/octet-stream",
    timeout: float = 5.0,
) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return r.status, r.read()


def sandbox_env() -> dict[str, str]:
    """Environment for spawned servers — adds repo root to PYTHONPATH."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env
