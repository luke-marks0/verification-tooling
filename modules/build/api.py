"""Deterministic / reproducible build — stable public API. See ``README.md``.

Two layers:
  * ``build_runtime`` — pure-Python lockfile enrichment (``cmd.builder``), runs
    anywhere (used by ``Pipeline.build``).
  * ``nix_build`` / ``build_oci`` / ``build_closure`` — drive the hermetic Nix
    build. These require Nix with flakes and shell out to ``nix build``.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from modules._cmd import REPO_ROOT, build_runtime

__all__ = ["build_runtime", "nix_build", "build_oci", "build_closure"]


def nix_build(attr: str, *, link: bool = False, cwd: Path | None = None) -> str:
    """Run ``nix build .#<attr>`` and return the store path.

    Requires Nix (flakes). ``link=False`` avoids writing a ``./result`` symlink.
    """
    args = ["nix", "build", f".#{attr}", "--print-out-paths", "--no-link" if not link else "--print-out-paths"]
    out = subprocess.run(
        args, cwd=str(cwd or REPO_ROOT), check=True, capture_output=True, text=True
    )
    return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""


def build_oci(**kwargs: Any) -> str:
    """Build the deterministic OCI image (``nix build .#oci``) -> store path."""
    return nix_build("oci", **kwargs)


def build_closure(**kwargs: Any) -> str:
    """Build the hermetic runtime closure (``nix build .#closure``) -> store path."""
    return nix_build("closure", **kwargs)
