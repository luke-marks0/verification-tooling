"""Smoke tests for the capability modules + recipe-book workflows.

All synthetic / no-GPU. Mirrors the proven d2 offline flow
(resolve -> build -> run x2 -> verify == conformant), but through the
``modules.Pipeline`` and ``workflows`` public surface.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules import Pipeline
from modules.network import egress_frames
from workflows.deterministic_inference_server import deterministic_inference_server
from workflows.deterministic_lora_training import assemble_plan

MANIFEST = "tests/fixtures/positive/manifest.v1.example.json"


class TestPipeline(unittest.TestCase):
    def test_synthetic_pipeline_is_conformant(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            pipe = Pipeline.from_manifest(MANIFEST).resolve().build()
            pipe.run(t / "a").run(t / "b")
            report = pipe.verify(report_out=t / "report.json", summary_out=t / "summary.txt")
            self.assertEqual(report["status"], "conformant")

    def test_verify_requires_two_runs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pipe = Pipeline.from_manifest(MANIFEST).resolve().build().run(Path(td) / "a")
            with self.assertRaises(ValueError):
                pipe.verify()


class TestNetworkFacade(unittest.TestCase):
    def test_egress_frames_reproducible(self) -> None:
        pipe = Pipeline.from_manifest(MANIFEST).resolve()
        payload = b"hello deterministic world"
        frames_a = egress_frames(payload, manifest=pipe.manifest, lockfile=pipe.lockfile)
        frames_b = egress_frames(payload, manifest=pipe.manifest, lockfile=pipe.lockfile)
        self.assertEqual(frames_a, frames_b)
        self.assertGreater(len(frames_a), 0)


class TestWorkflows(unittest.TestCase):
    def test_inference_server_workflow_synthetic(self) -> None:
        result = deterministic_inference_server(MANIFEST, mode="synthetic")
        self.assertEqual(result["status"], "conformant")
        self.assertTrue(result["frames_match"])
        self.assertGreater(result["frame_count"], 0)

    def test_lora_training_dry_run_is_deterministic(self) -> None:
        plan_a = assemble_plan(MANIFEST)
        plan_b = assemble_plan(MANIFEST)
        self.assertEqual(plan_a, plan_b)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG", plan_a["c3_env"])
        self.assertTrue(plan_a["runtime_closure_digest"].startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
