"""Compose deterministic-serving stages as a few readable Python lines.

The artifact spine ``manifest.v1 -> lockfile.v1 -> run_bundle.v1 ->
verify_report.v1`` already exists across ``cmd/{resolver,builder,runner,
verifier}``; :class:`Pipeline` wraps it so a workflow is a shareable file
instead of an ad-hoc bash script::

    report = (Pipeline.from_manifest("manifests/qwen3-1.7b.manifest.json")
              .resolve()             # -> lockfile.v1
              .build()               # -> closure digest
              .run("/tmp/a")         # -> run_bundle.v1
              .run("/tmp/b")         # -> run_bundle.v1 (independent run)
              .verify())             # -> verify_report.v1 ("conformant" iff identical)

``resolve``/``build`` pass dicts in memory; ``run``/``verify`` use the spine's
on-disk run bundles (the verifier compares bundle files).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modules import _cmd

_BUNDLE_NAME = "run_bundle.v1.json"


def _bundle_path(p: str | Path) -> Path:
    p = Path(p)
    return p / _BUNDLE_NAME if p.is_dir() else p


class Pipeline:
    """A composable build -> serve -> verify pipeline over one manifest."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.lockfile: dict[str, Any] | None = None
        self.bundles: list[Path] = []

    @classmethod
    def from_manifest(cls, path: str | Path) -> "Pipeline":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def resolve(self, **kwargs: Any) -> "Pipeline":
        """manifest -> lockfile (build determinism: resolve artifact digests)."""
        self.lockfile = _cmd.resolve_manifest_to_lockfile(self.manifest, **kwargs)
        return self

    def build(self, **kwargs: Any) -> "Pipeline":
        """lockfile -> lockfile + hermetic closure digest (build determinism)."""
        if self.lockfile is None:
            self.resolve()
        assert self.lockfile is not None
        self.lockfile = _cmd.build_runtime(self.lockfile, **kwargs)
        return self

    def run(self, out_dir: str | Path, *, mode: str = "synthetic", **kwargs: Any) -> "Pipeline":
        """One deterministic run -> ``out_dir/run_bundle.v1.json`` (inference determinism)."""
        if self.lockfile is None:
            self.build()
        assert self.lockfile is not None
        out = Path(out_dir)
        _cmd.run_inference(self.manifest, self.lockfile, out, mode=mode, **kwargs)
        self.bundles.append(out / _BUNDLE_NAME)
        return self

    def verify(
        self,
        baseline: str | Path | None = None,
        candidate: str | Path | None = None,
        *,
        report_out: str | Path | None = None,
        summary_out: str | Path | None = None,
    ) -> dict[str, Any]:
        """Compare two runs -> verify_report.v1. Defaults to the last two ``run()`` calls."""
        if baseline is None or candidate is None:
            if len(self.bundles) < 2:
                raise ValueError(
                    "verify() needs two runs; call .run() twice or pass baseline/candidate"
                )
            baseline = self.bundles[-2] if baseline is None else baseline
            candidate = self.bundles[-1] if candidate is None else candidate
        baseline = _bundle_path(baseline)
        candidate = _bundle_path(candidate)
        report_out = Path(report_out) if report_out else candidate.parent / "verify_report.json"
        summary_out = Path(summary_out) if summary_out else candidate.parent / "verify_summary.txt"
        return _cmd.verify_runs(baseline, candidate, report_out, summary_out)
