from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_cmd(args: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    # Run a bare "python3" with the *current* interpreter, so the suite works under
    # `.venv/bin/python3 -m unittest` without the venv being on PATH (the subprocess
    # would otherwise pick up a system python3 that lacks the project deps).
    if args and args[0] == "python3":
        args = [sys.executable, *args[1:]]
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)
    subprocess.run(args, check=True, env=merged_env, cwd=str(cwd) if cwd is not None else None)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n", encoding="utf-8")
