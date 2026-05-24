"""Builders for the placeholder Graph.

The real attested task graph (see the task-graph-prototype experiment on the
`experiments` branch) is the follow-up; for now the prover always returns the
empty-but-typed shape.
"""

from __future__ import annotations

from modules.core.common.deterministic import utc_now_iso
from modules.attestation.proverdet.wire import Graph


def build_empty_graph(run_id: str) -> Graph:
    """Empty placeholder graph for the given run_id.

    The verifier's /graph poll calls this; future work replaces the empty
    body with real task/artifact/transmission entries derived from the
    attested task graph.
    """
    return Graph(
        graph_version="v1-placeholder",
        run_id=run_id,
        produced_at=utc_now_iso(),
        tasks=[],
        artifacts=[],
        transmissions=[],
    )
