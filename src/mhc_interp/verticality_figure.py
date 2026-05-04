# %%
"""Verticality figure — show that load-bearing PT heads in mhc/mhc_lite have
heavy column-concentrated ('vertical') attention on AVERAGE, while pattern-
shaped PT heads have the canonical diagonal-stripe shape.

For each (model, criterion) ∈ {mhc, mhc_lite} × {top-N by pattern, top-N by
indirect ΔNLL}, average the attention matrices over the top-N heads. Then plot
the averaged attention as a heatmap, with a column-mean strip below to
quantify "verticality" (how concentrated each column is across query rows).

Output: results/analysis/verticality_pt_heads.png

Run:
    uv run python src/mhc_interp/verticality_figure.py
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

MODELS = ["mhc", "mhc_lite"]
MODEL_COLORS = {"mhc": "#c0392b", "mhc_lite": "#2980b9"}
TOP_N = 5


def load_attn(model: str) -> tuple[np.ndarray, list[str]]:
    A = np.load(HEADS_ROOT / "prev_token" / model / "attention.npy").astype(np.float32)
    tokens = json.loads((HEADS_ROOT / "prev_token" / model / "tokens.json").read_text())["tokens"]
    return A, tokens


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 4,  # 2 models × 4 panels (pat heatmap | pat col-mean | ind heatmap | ind col-mean)
        figsize=(15, 7),
        gridspec_kw={"hspace": 0.55, "wspace": 0.30,
                      "width_ratios": [1, 1, 1, 1]},
    )

    for ri, model in enumerate(MODELS):
        A, tokens = load_attn(model)
        T = A.shape[-1]
        df = pd.read_csv(HEADS_ROOT / "prev_token" / model / "path_patch.csv")
        top_pat = df.sort_values("pattern_score", ascending=False).iloc[:TOP_N]
        top_ind = df.sort_values("indirect_delta_nll", ascending=False).iloc[:TOP_N]

        # Average attention matrix per group.
        def avg_attn(rows):
            stack = np.stack([A[int(r["layer"]), int(r["head"])]
                              for _, r in rows.iterrows()])
            return stack.mean(axis=0)

        pat_avg = avg_attn(top_pat)
        ind_avg = avg_attn(top_ind)

        # ---- Pattern average heatmap
        ax = axes[ri, 0]
        ax.imshow(pat_avg, cmap="viridis", vmin=0, vmax=pat_avg.max(), aspect="equal")
        ax.set_xticks(range(T)); ax.set_yticks(range(T))
        ax.set_xticklabels([t.strip() or t for t in tokens],
                           rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels([t.strip() or t for t in tokens], fontsize=7)
        ax.set_title(f"{model}\navg(top-{TOP_N} by PATTERN)",
                     fontsize=11, fontweight="bold",
                     color=MODEL_COLORS[model], pad=4)
        if ri == 0:
            ax.set_xlabel("attended-to (key)", fontsize=8)

        # ---- Pattern column-mean line plot
        ax = axes[ri, 1]
        col_mean = pat_avg.mean(axis=0)
        ax.bar(range(T), col_mean, color=MODEL_COLORS[model], alpha=0.85,
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(T))
        ax.set_xticklabels([t.strip() or t for t in tokens],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("mean attention\n(column-wise)", fontsize=8.5)
        ax.set_title(f"column-mean of pattern-top-{TOP_N}", fontsize=10, pad=4)
        ax.set_ylim(0, max(0.4, col_mean.max() * 1.1))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        # ---- Indirect average heatmap
        ax = axes[ri, 2]
        ax.imshow(ind_avg, cmap="viridis", vmin=0, vmax=ind_avg.max(), aspect="equal")
        ax.set_xticks(range(T)); ax.set_yticks(range(T))
        ax.set_xticklabels([t.strip() or t for t in tokens],
                           rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels([t.strip() or t for t in tokens], fontsize=7)
        ax.set_title(f"{model}\navg(top-{TOP_N} by INDIRECT)",
                     fontsize=11, fontweight="bold",
                     color=MODEL_COLORS[model], pad=4)
        if ri == 0:
            ax.set_xlabel("attended-to (key)", fontsize=8)

        # ---- Indirect column-mean line plot
        ax = axes[ri, 3]
        col_mean = ind_avg.mean(axis=0)
        ax.bar(range(T), col_mean, color=MODEL_COLORS[model], alpha=0.85,
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(T))
        ax.set_xticklabels([t.strip() or t for t in tokens],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("mean attention\n(column-wise)", fontsize=8.5)
        ax.set_title(f"column-mean of indirect-top-{TOP_N}", fontsize=10, pad=4)
        ax.set_ylim(0, max(0.4, col_mean.max() * 1.1))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle(
        "Average attention shape: pattern-shaped vs load-bearing PT candidates",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig.text(
        0.5, -0.005,
        f"Each heatmap = mean attention over the top-{TOP_N} heads in that group. "
        "Diagonal stripe = canonical PT shape. Single bright COLUMN = attention sink / vertical. "
        "Bar charts (right of each heatmap) = column-wise mean attention; "
        "tall single bar = strong sink at that position.",
        ha="center", fontsize=8.5, style="italic", color="#444",
    )
    fig.tight_layout()
    out_path = OUT_DIR / "verticality_pt_heads.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    # Print verticality scores
    print("\nVerticality scores (max column-mean ÷ mean column-mean):")
    print("  > 2  = vertical/sink-like")
    print("  ~ 1  = uniform")
    for model in MODELS:
        A, _ = load_attn(model)
        df = pd.read_csv(HEADS_ROOT / "prev_token" / model / "path_patch.csv")
        for crit, label in [("pattern_score", "pattern"), ("indirect_delta_nll", "indirect")]:
            top = df.sort_values(crit, ascending=False).iloc[:TOP_N]
            stack = np.stack([A[int(r["layer"]), int(r["head"])]
                              for _, r in top.iterrows()])
            avg = stack.mean(axis=0)
            col_mean = avg.mean(axis=0)
            verticality = col_mean.max() / col_mean.mean()
            print(f"  {model:10s}  top-{TOP_N} by {label:8s}: verticality = {verticality:.2f}")


if __name__ == "__main__":
    main()

# %%
