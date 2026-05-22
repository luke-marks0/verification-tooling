"""Deterministic inference — stable public API. See ``README.md``.

Bitwise-deterministic vLLM serving via the "c3" config. Wraps the runner
(synthetic or vLLM) and the verifier from the artifact spine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from modules import _cmd

#: The "c3" determinism env vars. MUST be set before importing ``torch``/``vllm``.
#: The remaining two flags — ``enforce_eager=True`` and
#: ``attention_backend=FLASH_ATTN`` — are declared in the manifest ``runtime``.
C3_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "VLLM_BATCH_INVARIANT": "1",
    "PYTHONHASHSEED": "0",
}

__all__ = ["C3_ENV", "run_inference", "verify_runs"]


def run_inference(
    manifest: dict[str, Any],
    lockfile: dict[str, Any],
    out_dir: str | Path,
    *,
    mode: str = "synthetic",
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one deterministic inference pass -> ``out_dir/run_bundle.v1.json``.

    ``mode="synthetic"`` needs no GPU; ``mode="vllm"`` runs real inference.
    """
    return _cmd.run_inference(manifest, lockfile, Path(out_dir), mode=mode, **kwargs)


def verify_runs(
    baseline: str | Path,
    candidate: str | Path,
    *,
    report_out: str | Path,
    summary_out: str | Path,
) -> dict[str, Any]:
    """Compare two run bundles -> verify_report.v1 (``status == "conformant"`` iff identical)."""
    return _cmd.verify_runs(Path(baseline), Path(candidate), Path(report_out), Path(summary_out))
