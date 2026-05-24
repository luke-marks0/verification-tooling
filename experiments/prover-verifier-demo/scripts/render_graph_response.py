"""Render a picture of what the prover returns from GET /graph.

The current implementation always returns the empty placeholder; the picture
shows both the raw JSON response and the populated-shape the schema admits.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from modules.attestation.proverdet.graph_builder import build_empty_graph


def main() -> None:
    graph = build_empty_graph(run_id="demo-001")
    body = graph.model_dump(exclude_none=True)
    json_text = json.dumps(body, indent=2)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(
        "Prover GET /graph — current response (placeholder)",
        fontsize=14,
        fontweight="bold",
    )

    ax_json = fig.add_axes([0.05, 0.55, 0.42, 0.35])
    ax_json.axis("off")
    ax_json.set_title("Actual response body", loc="left", fontsize=11, fontweight="bold")
    ax_json.text(
        0.0,
        1.0,
        json_text,
        family="monospace",
        fontsize=10,
        va="top",
        ha="left",
    )

    ax_note = fig.add_axes([0.05, 0.08, 0.42, 0.40])
    ax_note.axis("off")
    ax_note.set_title("What this means", loc="left", fontsize=11, fontweight="bold")
    ax_note.text(
        0.0,
        1.0,
        (
            "tasks=[], artifacts=[], transmissions=[]\n\n"
            "The prover advertises only the metadata\n"
            "(graph_version, run_id, produced_at) and\n"
            "three empty arrays. There are no per-task\n"
            "claims to attribute against — which is why:\n\n"
            "  • compute_budget falls back to the\n"
            "    workload's /workload/stop summary, and\n"
            "  • bandwidth is a sums-comparison rather\n"
            "    than per-transmission attribution.\n\n"
            "Replacing this builder with the real attested\n"
            "task graph (experiments/task-graph-prototype/)\n"
            "is next-step #1 in the memo."
        ),
        family="sans-serif",
        fontsize=10,
        va="top",
        ha="left",
    )

    ax_schema = fig.add_axes([0.52, 0.08, 0.45, 0.82])
    ax_schema.axis("off")
    ax_schema.set_xlim(0, 10)
    ax_schema.set_ylim(0, 10)
    ax_schema.set_title(
        "Schema shape (what populated entries would look like)",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )

    boxes = [
        (
            "Task",
            (
                "task_id: str\n"
                "pod_id: str\n"
                "operation: str\n"
                "claimed_flops: int\n"
            ),
            7.0,
            "#cfe8ff",
        ),
        (
            "Artifact",
            (
                "artifact_id: str\n"
                "commitment: sha256:…\n"
                "size_bytes: int\n"
            ),
            4.4,
            "#d4f1d4",
        ),
        (
            "Transmission",
            (
                "transmission_id: str\n"
                "sender_pod_id: str\n"
                "receiver_pod_id: str\n"
                "artifact_id: str  (FK)\n"
                "tap_signature: hex\n"
            ),
            1.4,
            "#ffe4b3",
        ),
    ]
    for title, body_text, y, color in boxes:
        box = FancyBboxPatch(
            (0.5, y),
            9.0,
            2.2,
            boxstyle="round,pad=0.05",
            linewidth=1.0,
            edgecolor="#444",
            facecolor=color,
        )
        ax_schema.add_patch(box)
        ax_schema.text(
            0.7,
            y + 1.95,
            title,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
        )
        ax_schema.text(
            0.7,
            y + 1.55,
            body_text,
            fontsize=9,
            family="monospace",
            va="top",
            ha="left",
        )

    ax_schema.annotate(
        "currently empty",
        xy=(9.4, 9.3),
        xytext=(7.0, 9.6),
        fontsize=9,
        color="#a33",
        fontweight="bold",
        ha="right",
    )

    out = Path(__file__).resolve().parents[1] / "figures" / "graph_response.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
