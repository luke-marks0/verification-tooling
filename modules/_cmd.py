"""Internal glue: load the ``cmd/*`` pipeline-stage functions in-memory.

``cmd/*/main.py`` are standalone scripts (no ``__init__.py``), so we load them
by file path with ``importlib`` rather than importing ``cmd`` as a package.
This lets the capability modules and the :class:`~modules.pipeline.Pipeline`
compose the stages as plain Python calls (dict in / dict out) instead of
shelling out — see ``docs/plans/repo-modularization.md``.

This is the one place that knows how the runner's provenance/pod arguments are
defaulted; everything else composes through here.
"""
from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@lru_cache(maxsize=None)
def _load(rel_path: str, mod_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(mod_name, REPO_ROOT / rel_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {rel_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_manifest_to_lockfile(manifest: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """manifest.v1 dict -> lockfile.v1 dict (cmd/resolver)."""
    fn = _load("cmd/resolver/main.py", "_dss_resolver").resolve_manifest_to_lockfile
    return fn(manifest, **kwargs)


def build_runtime(lockfile: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """lockfile.v1 dict -> lockfile.v1 dict enriched with the build closure (cmd/builder)."""
    fn = _load("cmd/builder/main.py", "_dss_builder").build_runtime
    return fn(lockfile, **kwargs)


def run_inference(
    manifest: dict[str, Any],
    lockfile: dict[str, Any],
    out_dir: Path,
    *,
    replica_id: str = "replica-0",
    mode: str = "synthetic",
    network_backend: str = "sim",
    runtime_hardware: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one pass -> writes ``out_dir/run_bundle.v1.json`` and returns the bundle (cmd/runner).

    Provenance/pod arguments mirror ``cmd/runner/main.py``'s ``main()`` defaults.
    ``invocation_argv`` is deliberately independent of ``out_dir`` so two runs
    differing only in output directory remain bitwise-comparable.
    """
    run = _load("cmd/runner/main.py", "_dss_runner").run
    return run(
        manifest,
        lockfile,
        Path(out_dir),
        replica_id,
        mode=mode,
        network_backend=network_backend,
        runtime_hardware=runtime_hardware,
        pod_manifest_path=kwargs.pop("pod_manifest_path", "manifest.json"),
        pod_lockfile_path=kwargs.pop("pod_lockfile_path", "lockfile.json"),
        pod_runtime_closure_path=kwargs.pop(
            "pod_runtime_closure_path", "lockfile.runtime_closure_digest"
        ),
        pod_name=kwargs.pop("pod_name", "local-pod"),
        node_name=kwargs.pop("node_name", "local-node"),
        namespace=kwargs.pop("namespace", "default"),
        invocation_argv=kwargs.pop("invocation_argv", ["modules.pipeline", f"mode={mode}"]),
        **kwargs,
    )


def verify_runs(
    baseline_bundle: Path,
    candidate_bundle: Path,
    report_out: Path,
    summary_out: Path,
) -> dict[str, Any]:
    """Compare two run bundles -> verify_report.v1 dict (cmd/verifier)."""
    verify = _load("cmd/verifier/main.py", "_dss_verifier").verify
    return verify(Path(baseline_bundle), Path(candidate_bundle), Path(report_out), Path(summary_out))
