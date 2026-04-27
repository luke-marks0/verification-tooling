#!/usr/bin/env python3
"""Generate updated overhead figures from the full vLLM 0.19.1 sweep data.

Mirrors the original overhead-benchmark plots but with new configs:
  baseline: no determinism flags (CUDA Graphs + torch.compile on)
  boi:      VLLM_BATCH_INVARIANT=1 + FLASH_ATTN (deterministic, graphs on)
  all:      boi + CUBLAS_WORKSPACE_CONFIG (identical to boi)
  eager:    enforce_eager + all flags (deterministic, graphs off)
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

EXPERIMENT = pathlib.Path(__file__).resolve().parents[1]
DATA = EXPERIMENT / "data" / "sweep.jsonl"
OUT = EXPERIMENT / "figures"
OUT.mkdir(exist_ok=True)

rows = [json.loads(l) for l in DATA.read_text().splitlines()]

MODELS = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen 2.5 1.5B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B",
}
CONFIGS = ["baseline", "boi", "eager"]  # skip "all" since it's identical to boi
CONFIG_LABELS = {
    "baseline": "Baseline (no determinism)",
    "boi": "BOI + FLASH_ATTN (graphs on)",
    "all": "BOI + CUBLAS (graphs on)",
    "eager": "enforce_eager (graphs off)",
}
BATCH_SIZES = [1, 4, 16, 64, 128]
SEQ_LENS = [16, 128, 512, 2048]

lookup = {}
for r in rows:
    key = (r["model"], r["config"], r["batch_size"], r["max_tokens"])
    lookup[key] = r["tok_per_s"]


# ── Figure 1: Throughput by batch size (mirroring original) ────────────
def plot_throughput_by_batch(seq_len=128):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    colors = {"baseline": "#2196F3", "boi": "#4CAF50", "eager": "#F44336"}
    markers = {"baseline": "o", "boi": "s", "eager": "^"}

    for ax, (model_id, model_name) in zip(axes, MODELS.items()):
        for cfg in CONFIGS:
            throughputs = [lookup.get((model_id, cfg, bs, seq_len), 0)
                          for bs in BATCH_SIZES]
            ax.plot(range(len(BATCH_SIZES)), throughputs,
                    f"{markers[cfg]}-", color=colors[cfg],
                    label=CONFIG_LABELS[cfg],
                    linewidth=2, markersize=7)

        ax.set_xticks(range(len(BATCH_SIZES)))
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Throughput (tokens/s)")
        ax.set_title(model_name)
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    fig.suptitle(f"Throughput vs. Determinism Config — vLLM 0.19.1 (seq_len={seq_len})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "throughput_by_batch.png", dpi=200, bbox_inches="tight")
    print(f"Saved {OUT / 'throughput_by_batch.png'}")
    plt.close()


# ── Figure 2: Overhead % comparing BOI and eager to baseline ───────────
def plot_overhead_pct():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, (model_id, model_name) in zip(axes, MODELS.items()):
        for cfg, color, marker, ls in [
            ("boi", "#4CAF50", "s", "-"),
            ("eager", "#F44336", "^", "--"),
        ]:
            overheads = []
            for bs in BATCH_SIZES:
                vals = []
                for sl in SEQ_LENS:
                    c0 = lookup.get((model_id, "baseline", bs, sl), 1)
                    cn = lookup.get((model_id, cfg, bs, sl), 0)
                    vals.append((cn / c0 - 1) * 100)
                overheads.append(np.mean(vals))
            ax.plot(range(len(BATCH_SIZES)), overheads,
                    f"{marker}{ls}", color=color, label=CONFIG_LABELS[cfg],
                    linewidth=2, markersize=7)

        ax.set_xticks(range(len(BATCH_SIZES)))
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Throughput change vs baseline (%)")
        ax.set_title(model_name)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle("Determinism Overhead: BOI (graphs on) vs enforce_eager (graphs off)\nvLLM 0.19.1, averaged across seq lengths",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "overhead_pct.png", dpi=200, bbox_inches="tight")
    print(f"Saved {OUT / 'overhead_pct.png'}")
    plt.close()


# ── Figure 3: Speedup of BOI over eager ────────────────────────────────
def plot_boi_vs_eager_speedup():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    colors = ["#9C27B0", "#E91E63", "#FF5722", "#795548"]

    for ax, (model_id, model_name) in zip(axes, MODELS.items()):
        for si, sl in enumerate(SEQ_LENS):
            speedups = []
            for bs in BATCH_SIZES:
                boi = lookup.get((model_id, "boi", bs, sl), 1)
                eager = lookup.get((model_id, "eager", bs, sl), 1)
                speedups.append(boi / eager if eager > 0 else 1)
            ax.plot(range(len(BATCH_SIZES)), speedups,
                    "o-", color=colors[si], label=f"seq={sl}",
                    linewidth=2, markersize=6)

        ax.set_xticks(range(len(BATCH_SIZES)))
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Speedup (BOI / eager)")
        ax.set_title(model_name)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(1, color="black", linewidth=0.5, linestyle="--")

    fig.suptitle("Speedup from Keeping CUDA Graphs (BOI vs enforce_eager)\nvLLM 0.19.1 — both deterministic, graphs make the difference",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "boi_vs_eager_speedup.png", dpi=200, bbox_inches="tight")
    print(f"Saved {OUT / 'boi_vs_eager_speedup.png'}")
    plt.close()


# ── Figure 4: Side-by-side bar — the money chart ──────────────────────
def plot_summary_bars():
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(BATCH_SIZES))
    width = 0.35
    model_styles = [
        ("#4CAF50", "#1B5E20"),
        ("#1E88E5", "#0D47A1"),
    ]

    for i, ((model_id, model_name), (color, edge)) in enumerate(
        zip(MODELS.items(), model_styles)
    ):
        boi_by_batch = []
        for bs in BATCH_SIZES:
            vals = []
            for sl in SEQ_LENS:
                c0 = lookup.get((model_id, "baseline", bs, sl), 1)
                boi = lookup.get((model_id, "boi", bs, sl), 0)
                vals.append((1 - boi / c0) * 100)
            boi_by_batch.append(np.mean(vals))

        offset = (i - 0.5) * width
        ax.bar(x + offset, boi_by_batch, width,
               color=color, edgecolor=edge, linewidth=1,
               label=model_name)

    ax.set_xticks(x)
    ax.set_xticklabels(BATCH_SIZES)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput lost vs baseline (%)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(OUT / "summary_overhead_comparison.png", dpi=200, bbox_inches="tight")
    print(f"Saved {OUT / 'summary_overhead_comparison.png'}")
    plt.close()


if __name__ == "__main__":
    plot_throughput_by_batch()
    plot_overhead_pct()
    plot_boi_vs_eager_speedup()
    plot_summary_bars()
    print("Done.")
