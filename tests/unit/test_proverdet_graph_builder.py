from __future__ import annotations

import unittest

from modules.core.common.contracts import validate_with_schema
from modules.attestation.proverdet.graph_builder import build_empty_graph
from modules.attestation.proverdet.wire import Graph


class TestBuildEmptyGraph(unittest.TestCase):
    def test_returns_a_graph_instance(self) -> None:
        g = build_empty_graph(run_id="abc")
        self.assertIsInstance(g, Graph)

    def test_has_no_tasks_artifacts_or_transmissions(self) -> None:
        g = build_empty_graph(run_id="abc")
        self.assertEqual(g.tasks, [])
        self.assertEqual(g.artifacts, [])
        self.assertEqual(g.transmissions, [])

    def test_run_id_is_preserved(self) -> None:
        g = build_empty_graph(run_id="my-run-001")
        self.assertEqual(g.run_id, "my-run-001")

    def test_graph_version_is_v1_placeholder(self) -> None:
        g = build_empty_graph(run_id="abc")
        self.assertEqual(g.graph_version, "v1-placeholder")

    def test_validates_against_schema(self) -> None:
        g = build_empty_graph(run_id="abc")
        validate_with_schema("prover_graph.v1.schema.json", g.model_dump(exclude_none=True))

    def test_produced_at_is_iso_z(self) -> None:
        g = build_empty_graph(run_id="abc")
        self.assertTrue(g.produced_at.endswith("Z"))


if __name__ == "__main__":
    unittest.main()
