"""Memory wipe + erasure attestation (PoSE). See ``README.md``.

The PoSE implementation lives as a regular sub-package at ``modules/memory/pose``.
The heavy backends — HBM via CUDA, AES noise — import lazily, so importing
this facade never needs a GPU.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

#: Directory holding the ``pose`` package source.
POSE_SRC = Path(__file__).resolve().parent / "pose"

__all__ = ["POSE_SRC", "load_pose"]


def load_pose(submodule: str = "") -> ModuleType:
    """Import the ``pose`` package, or a submodule (e.g. ``"protocol"``,
    ``"prover"``, ``"memory.dram"``).

    Lazy by design: backends that need the target box's environment (HBM/CUDA,
    AES noise) are only imported when requested.
    """
    name = "modules.memory.pose" + (f".{submodule}" if submodule else "")
    return importlib.import_module(name)
