# %%
"""Top-10 head inventory across the 3 mhc-781m variants.

Single composite figure: one row per detector, one column per model. Each
cell shows the top-10 ranked heads as a horizontal bar chart with their
(layer, head) label and detector score. Layer numbers are color-shaded so
the reader can also see depth at a glance.

Output:
  results/analysis/top10_per_detector.png
  results/analysis/top10_per_detector.csv

Run:
    uv run python src/mhc_interp/top10_inventory.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

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

TOP_K = 10
N_LAYERS = 36

# Layer-depth color ramp: early = light, late = dark.
LAYER_CMAP = LinearSegmentedColormap.from_list(
    "depth", ["#fde0dd", "#fa9fb5", "#c51b8a", "#7a0177", "#3d004f"]
)


def load_top10(slug: str, model: str) -> list[dict]:
    return json.loads((HEADS_ROOT / slug / model / "top_heads.json").read_text())[:TOP_K]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build the long-form CSV first.
    rows = []
    for slug in DETECTORS:
        for model in MODELS:
            for entry in load_top10(slug, model):
                rows.append({
                    "head_type": slug, "model": model,
                    "rank": entry["rank"], "layer": entry["layer"],
                    "head": entry["head"], "score": entry["score"],
                })
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "top10_per_detector.csv"
    df.to_csv(csv_path, index=False)
    print(f"saved {csv_path}  ({len(df)} rows)")

    n_rows = len(DETECTORS)
    n_cols = len(MODELS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(15, 18),
        gridspec_kw={"hspace": 0.55, "wspace": 0.35},
    )

    # Compute a shared x-range per detector so the three model panels line up.
    det_max_score = {d: max(load_top10(d, m)[0]["score"] for m in MODELS) for d in DETECTORS}
    det_min_score = {
        d: min(min(e["score"] for e in load_top10(d, m)) for m in MODELS) for d in DETECTORS
    }

    for ri, slug in enumerate(DETECTORS):
        for ci, model in enumerate(MODELS):
            ax = axes[ri, ci]
            top = load_top10(slug, model)
            labels = [f"L{e['layer']:2d}  h{e['head']:2d}" for e in top]
            scores = [e["score"] for e in top]
            layers = [e["layer"] for e in top]

            colors = [LAYER_CMAP(li / (N_LAYERS - 1)) for li in layers]
            y_positions = np.arange(len(top))
            ax.barh(y_positions, scores, color=colors, edgecolor="black", linewidth=0.5)
            for y, lbl, s in zip(y_positions, labels, scores):
                # Score on the right of each bar.
                offset = 0.02 * max(abs(det_max_score[slug]), abs(det_min_score[slug]) + 0.05)
                ha = "left" if s >= 0 else "right"
                x = s + (offset if s >= 0 else -offset)
                txt = f"{s:.2f}" if s >= 0 else f"{s:+.2f}"
                ax.text(x, y, txt, va="center", ha=ha, fontsize=8.5, fontweight="bold")
            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels, fontsize=8.5, fontfamily="monospace")
            ax.invert_yaxis()
            xmin = min(det_min_score[slug] - 0.05, 0)
            xmax = det_max_score[slug] * 1.25
            ax.set_xlim(xmin, xmax)
            ax.axvline(0, color="black", lw=0.5)
            ax.tick_params(axis="x", labelsize=8)

            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

            # Title
            if ri == 0:
                ax.set_title(model, fontsize=12, fontweight="bold",
                             color=MODEL_COLORS[model], pad=8)
            if ci == 0:
                ax.set_ylabel(DET_LABELS[slug], fontsize=12, fontweight="bold",
                              labelpad=14)

    # Layer-depth colorbar at the bottom.
    cbar_ax = fig.add_axes([0.18, 0.04, 0.65, 0.012])
    sm = plt.cm.ScalarMappable(
        cmap=LAYER_CMAP, norm=plt.Normalize(vmin=0, vmax=N_LAYERS - 1)
    )
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("layer index (bar color)", fontsize=10)
    cbar.set_ticks([0, 5, 10, 15, 20, 25, 30, 35])

    fig.suptitle(
        f"Top-{TOP_K} heads per detector × model  —  bar = score, color = layer depth",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.025,
        "Each panel shows the top-10 (layer, head) pairs ranked by detector score "
        "(highest at top). Bar fill encodes layer (light = early, dark = late).",
        ha="center", fontsize=9.5, style="italic", color="#444",
    )

    out_path = OUT_DIR / "top10_per_detector.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

# %%
