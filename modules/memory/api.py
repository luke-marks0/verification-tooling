"""Memory wipe + erasure attestation (PoSE) — facade. See ``README.md``.

The PoSE implementation is a *separately-deployed* package (``pose``) under
``experiments/memory_wipe/src``, installed on the target GPU box via ``uv``.
This facade makes it importable from the main tree **without relocating the
deployable package** (a physical move would break the remote uv-install
workflow, which can't be verified in CI). The heavy interfaces — HBM via CUDA,
AES noise — import lazily, so importing this facade never needs a GPU.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

from modules._cmd import REPO_ROOT

#: Directory holding the ``pose`` package source.
POSE_SRC = REPO_ROOT / "experiments" / "memory_wipe" / "src"

__all__ = ["POSE_SRC", "ensure_pose_on_path", "load_pose"]


def ensure_pose_on_path() -> Path:
    """Put the ``pose`` package source on ``sys.path``. Returns its directory."""
    if POSE_SRC.is_dir() and str(POSE_SRC) not in sys.path:
        sys.path.insert(0, str(POSE_SRC))
    return POSE_SRC


def load_pose(submodule: str = "") -> ModuleType:
    """Import the ``pose`` package, or a submodule (e.g. ``"protocol"``,
    ``"prover"``, ``"memory.dram"``).

    Lazy by design: backends that need the target box's environment (HBM/CUDA,
    AES noise) are only imported when requested.
    """
    ensure_pose_on_path()
    name = "pose" + (f".{submodule}" if submodule else "")
    return importlib.import_module(name)
