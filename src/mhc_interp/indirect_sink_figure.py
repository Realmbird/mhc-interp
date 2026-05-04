# %%
"""Standalone evidence: load-bearing PT candidates in mhc / mhc_lite are
attention-sink heads, NOT prev-token heads.

For each model:
  * top-5 individual attention thumbnails (indirect-top heads)
  * mean attention over those 5 heads
  * column-mean attention bar chart with per-head values plotted as dots —
    so we can see that each individual head sinks to the same position,
    not just the average.

This is meant to defend the claim independently — no pattern-detector
comparison panel.

Output:
  results/analysis/indirect_pt_sink.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HEADS_ROOT = REPO_ROOT / "results" / "heads"
OUT_DIR = REPO_ROOT / "results" / "analysis"

MODELS = ["mhc", "mhc_lite"]
MODEL_COLORS = {"mhc": "#c0392b", "mhc_lite": "#2980b9"}
TOP_N = 5


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 9))
    # Layout: 2 rows × (TOP_N + 2) cols. Bottom of each row also has a
    # column-mean strip; achieved by stacking another GridSpec row per model.
    n_cols = TOP_N + 2  # 5 individuals + average + col-mean strip
    outer_gs = GridSpec(
        4, n_cols, figure=fig,
        height_ratios=[1.2, 0.6, 1.2, 0.6],  # head, col-mean, head, col-mean
        hspace=0.55, wspace=0.30,
    )

    for ri, model in enumerate(MODELS):
        # Load data
        A = np.load(HEADS_ROOT / "prev_token" / model / "attention.npy").astype(np.float32)
        tokens = json.loads((HEADS_ROOT / "prev_token" / model / "tokens.json").read_text())["tokens"]
        T = A.shape[-1]
        df = pd.read_csv(HEADS_ROOT / "prev_token" / model / "path_patch.csv")
        top_ind = df.sort_values("indirect_delta_nll", ascending=False).iloc[:TOP_N]

        # Stack the top-N attention matrices.
        stack = np.stack([A[int(r["layer"]), int(r["head"])]
                          for _, r in top_ind.iterrows()])  # (TOP_N, T, T)
        avg_attn = stack.mean(axis=0)
        per_head_col_mean = stack.mean(axis=1)  # (TOP_N, T)
        avg_col_mean = avg_attn.mean(axis=0)    # (T,)

        head_row = ri * 2
        bar_row = ri * 2 + 1

        # ---- Per-head thumbnails (TOP_N panels)
        for ci, (_, r) in enumerate(top_ind.iterrows()):
            ax = fig.add_subplot(outer_gs[head_row, ci])
            li = int(r["layer"]); hi = int(r["head"])
            ind = float(r["indirect_delta_nll"])
            ax.imshow(stack[ci], cmap="viridis", vmin=0, vmax=1, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"L{li}h{hi}\nind={ind:+.3f}",
                         fontsize=9.5, color=MODEL_COLORS[model],
                         fontweight="bold", pad=4)
            if ci == 0:
                ax.set_ylabel(model, fontsize=12, fontweight="bold",
                              color=MODEL_COLORS[model], labelpad=12)

        # ---- Averaged thumbnail
        ax = fig.add_subplot(outer_gs[head_row, TOP_N])
        ax.imshow(avg_attn, cmap="viridis", vmin=0,
                  vmax=avg_attn.max(), aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"average\n(top-{TOP_N})",
                     fontsize=10, fontweight="bold", pad=4)

        # ---- Verticality stat panel
        ax = fig.add_subplot(outer_gs[head_row, TOP_N + 1])
        verticality = avg_col_mean.max() / avg_col_mean.mean()
        max_col = int(avg_col_mean.argmax())
        attn_pos0 = float(avg_attn[:, 0].mean())
        ax.axis("off")
        ax.text(0.5, 0.55,
                f"verticality\n= {verticality:.2f}",
                ha="center", va="center", fontsize=14, fontweight="bold",
                transform=ax.transAxes, color=MODEL_COLORS[model])
        ax.text(0.5, 0.18,
                f"sink at\n'{tokens[max_col].strip() or tokens[max_col]}' (col {max_col})\n"
                f"avg attn → pos 0: {attn_pos0:.2f}",
                ha="center", va="center", fontsize=9.5,
                transform=ax.transAxes, color="#444")

        # ---- Column-mean bar chart with per-head dots overlaid (full row width)
        ax = fig.add_subplot(outer_gs[bar_row, :])
        x = np.arange(T)
        # Average column-mean as the bars
        ax.bar(x, avg_col_mean, color=MODEL_COLORS[model], alpha=0.5,
               edgecolor="black", linewidth=0.5,
               label=f"average column-mean (top-{TOP_N})")
        # Per-head dots
        rng = np.random.default_rng(0)
        for hi in range(TOP_N):
            jitter = rng.uniform(-0.18, 0.18, size=T)
            ax.scatter(x + jitter, per_head_col_mean[hi],
                       s=22, color="black", alpha=0.7, zorder=3,
                       edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([t.strip() or t for t in tokens],
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("column-mean attention", fontsize=10)
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(0, max(0.4, per_head_col_mean.max() * 1.05))
        ax.legend(fontsize=9, loc="upper right", frameon=False)
        # add small annotation: each dot is one of the 5 heads
        ax.text(0.005, 0.97, f"black dots = each of the {TOP_N} individual heads",
                transform=ax.transAxes, fontsize=8.5, color="#555",
                style="italic", va="top")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle(
        "Load-bearing prev-token candidates in MHC are attention-sink heads",
        fontsize=14, fontweight="bold", y=1.00,
    )
    fig.text(
        0.5, -0.005,
        "Top-5 heads per model ranked by indirect ΔNLL on the prev-token probe. "
        "Each head's attention sinks to position 0 ('When'), confirmed by the column-mean bar plus the "
        "per-head dot overlay (dots cluster at the same column the average peaks at). "
        "Verticality = max(column-mean) ÷ mean(column-mean); >2 = sink-like.",
        ha="center", fontsize=8.5, style="italic", color="#444",
    )
    fig.tight_layout()
    out_path = OUT_DIR / "indirect_pt_sink.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

# %%
