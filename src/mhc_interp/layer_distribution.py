# %%
"""Box-and-whisker plot of the layer index where each attention-head type
concentrates, per model.

For each (head_type, model) we take the top-K heads from
results/heads/{slug}/{model}/top_heads.json and render their layer numbers as
a boxplot. Three boxes per head type (one per model), so the user can read off
"induction heads sit at layers 7-9 in MHC vs 18-21 in residual" at a glance.

Output:
  results/analysis/layer_distribution_boxplot.png
  results/analysis/layer_distribution_top{K}.csv

Run:
    uv run python src/mhc_interp/layer_distribution.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HEADS_ROOT = REPO_ROOT / "results" / "heads"
OUT_DIR = REPO_ROOT / "results" / "analysis"

DETECTORS = ["prev_token", "induction", "duplicate", "successor", "copy_suppression"]
DET_LABELS = {
    "prev_token": "prev-token",
    "induction": "induction",
    "duplicate": "duplicate",
    "successor": "successor",
    "copy_suppression": "copy-suppression",
}
MODELS = ["residual", "mhc", "mhc_lite"]
MODEL_COLORS = {"residual": "#7a7a7a", "mhc": "#c0392b", "mhc_lite": "#2980b9"}

# How many top heads to pull from each (slug, model). K=20 ≈ top 2.8% of 720.
TOP_K = 20
N_LAYERS = 36  # all 3 variants are GPT-2-large shape


def collect_layers() -> pd.DataFrame:
    rows = []
    for slug in DETECTORS:
        for model in MODELS:
            top_path = HEADS_ROOT / slug / model / "top_heads.json"
            top = json.loads(top_path.read_text())
            for entry in top[:TOP_K]:
                rows.append({
                    "head_type": slug,
                    "model": model,
                    "rank": entry["rank"],
                    "layer": entry["layer"],
                    "head": entry["head"],
                    "score": entry["score"],
                })
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = collect_layers()
    csv_path = OUT_DIR / f"layer_distribution_top{TOP_K}.csv"
    df.to_csv(csv_path, index=False)
    print(f"saved {csv_path}  ({len(df)} rows)")

    fig, ax = plt.subplots(figsize=(13, 6.5))

    n_dets = len(DETECTORS)
    n_models = len(MODELS)
    group_centers = np.arange(n_dets)
    box_width = 0.22
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * (box_width + 0.04)

    medians = {}  # for annotation later

    for mi, model in enumerate(MODELS):
        positions = group_centers + offsets[mi]
        data = []
        for slug in DETECTORS:
            sub = df[(df["head_type"] == slug) & (df["model"] == model)]
            data.append(sub["layer"].to_numpy())
            medians[(slug, model)] = float(np.median(sub["layer"]))

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white",
                       "markeredgecolor": "black", "markersize": 5},
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"facecolor": MODEL_COLORS[model], "alpha": 0.75,
                      "edgecolor": "black"},
            whiskerprops={"color": "black"},
            capprops={"color": "black"},
            flierprops={"marker": "o", "markersize": 3.5,
                        "markerfacecolor": MODEL_COLORS[model],
                        "markeredgecolor": "black", "alpha": 0.6},
        )
        # Scatter the actual top-K layer values to give density a visible feel.
        rng = np.random.default_rng(42 + mi)
        for x, ys in zip(positions, data):
            jitter = rng.uniform(-1, 1, size=len(ys)) * box_width / 3
            ax.scatter(np.full(len(ys), x) + jitter, ys, s=8,
                       color="black", alpha=0.45, zorder=3)

    # Median annotations above each box, color-matched.
    for di, slug in enumerate(DETECTORS):
        for mi, model in enumerate(MODELS):
            x = group_centers[di] + offsets[mi]
            med = medians[(slug, model)]
            ax.text(x, N_LAYERS + 1.2, f"L{int(med)}",
                    ha="center", va="bottom", fontsize=8.5,
                    color=MODEL_COLORS[model], fontweight="bold")

    ax.set_xticks(group_centers)
    ax.set_xticklabels([DET_LABELS[d] for d in DETECTORS], fontsize=10.5)
    ax.set_ylabel("Layer index (0 = embed → 35 = pre-unembed)", fontsize=11)
    ax.set_ylim(-1, N_LAYERS + 4.5)
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30, 35])
    ax.invert_yaxis()  # layer 0 at the top reads more naturally for "where in depth"
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m], alpha=0.75) for m in MODELS]
    ax.legend(handles, MODELS, loc="lower right", title="model", framealpha=0.95)

    ax.set_title(
        f"Layer distribution of top-{TOP_K} heads per detector  —  median annotated above each box",
        fontsize=13, fontweight="bold",
    )
    fig.text(
        0.5, -0.015,
        "Each box = top-20 heads (out of 720) ranked by detector score; "
        "white diamond = mean, black line = median, dots = individual head layers (jittered).",
        ha="center", fontsize=8.5, style="italic", color="#444",
    )

    out_path = OUT_DIR / "layer_distribution_boxplot.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    # Print medians table for the writeup.
    print("\nMedian layer per (head_type, model):")
    pivot = df.groupby(["head_type", "model"])["layer"].median().unstack()
    print(pivot.reindex(DETECTORS)[MODELS].astype(int).to_string())


if __name__ == "__main__":
    main()

# %%
