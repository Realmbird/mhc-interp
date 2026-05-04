# %%
"""Visualize pattern-detection vs ablation-detection agreement for the two
pattern detectors that have ablation data (induction, prev_token) on the two
MHC variants.

For each (detector, model) we plot all 720 heads as one point in
(pattern_score, ablation_delta_nll) space. Top-10 heads by each axis are
highlighted; intersections are emphasized.

Output:
  results/analysis/pattern_vs_ablation_match.png
  results/analysis/pattern_vs_ablation_match.csv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HEADS_ROOT = REPO_ROOT / "results" / "heads"
OUT_DIR = REPO_ROOT / "results" / "analysis"

DETECTORS = [
    ("induction", "Induction"),
    ("prev_token", "Previous-token"),
]
MODELS = ["mhc", "mhc_lite"]
MODEL_COLORS = {"mhc": "#c0392b", "mhc_lite": "#2980b9"}
TOP_K = 10


def load_pair(slug: str, model: str) -> pd.DataFrame:
    df = pd.read_csv(HEADS_ROOT / slug / model / "scores.csv")
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    summary_rows = []
    for ri, (slug, det_label) in enumerate(DETECTORS):
        for ci, model in enumerate(MODELS):
            ax = axes[ri, ci]
            df = load_pair(slug, model)
            pat = df["pattern_score"].to_numpy()
            abl = df["ablation_delta_nll"].to_numpy()

            # Rank arrays for top-K membership.
            pat_rank = pd.Series(pat).rank(ascending=False).to_numpy()
            abl_rank = pd.Series(abl).rank(ascending=False).to_numpy()

            top_p = pat_rank <= TOP_K
            top_a = abl_rank <= TOP_K
            both = top_p & top_a
            only_p = top_p & ~top_a
            only_a = top_a & ~top_p
            neither = ~(top_p | top_a)

            # Background scatter: rest of heads in light gray.
            ax.scatter(pat[neither], abl[neither], s=10, color="#bbbbbb",
                       alpha=0.45, edgecolor="none")
            # Top-10 by pattern only — pink-ringed.
            ax.scatter(pat[only_p], abl[only_p], s=70, color="white",
                       edgecolor="#e67e22", linewidth=1.6,
                       label=f"top-{TOP_K} pattern only")
            # Top-10 by ablation only — blue-ringed.
            ax.scatter(pat[only_a], abl[only_a], s=70, color="white",
                       edgecolor="#2980b9", linewidth=1.6,
                       label=f"top-{TOP_K} ablation only")
            # Intersection — solid model color, big.
            ax.scatter(pat[both], abl[both], s=100, color=MODEL_COLORS[model],
                       edgecolor="black", linewidth=1.0,
                       label=f"top-{TOP_K} BOTH ({both.sum()})")

            # Annotate the intersection points with (L, h).
            for idx in np.where(both)[0]:
                li = int(df["layer"].iloc[idx])
                hi = int(df["head"].iloc[idx])
                ax.annotate(f"L{li}h{hi}", (pat[idx], abl[idx]),
                            textcoords="offset points",
                            xytext=(6, 4), fontsize=8, fontweight="bold")

            # Stats
            pearson = float(np.corrcoef(pat, abl)[0, 1])
            top10_overlap = int(both.sum())
            top10_x_top20 = int((top_p & (abl_rank <= 20)).sum())
            top10_x_top50 = int((top_p & (abl_rank <= 50)).sum())
            summary_rows.append({
                "detector": slug, "model": model,
                "pearson": pearson,
                "top10_x_top10": top10_overlap,
                "top10_x_top20": top10_x_top20,
                "top10_x_top50": top10_x_top50,
            })

            # Annotation box: stats inside the panel
            stats_txt = (
                f"Pearson r = {pearson:+.3f}\n"
                f"top-10 ∩ top-10 = {top10_overlap}\n"
                f"top-10 ∩ top-20 = {top10_x_top20}\n"
                f"top-10 ∩ top-50 = {top10_x_top50}"
            )
            ax.text(0.02, 0.98, stats_txt, transform=ax.transAxes,
                    ha="left", va="top", fontsize=9, family="monospace",
                    bbox=dict(facecolor="white", edgecolor="#888",
                              alpha=0.9, pad=4))

            ax.set_xlabel("pattern score (canonical-stripe mass)", fontsize=10)
            ax.set_ylabel("ablation Δ NLL (nats)", fontsize=10)
            ax.set_title(f"{det_label}  ·  {model}",
                         fontsize=12, fontweight="bold",
                         color=MODEL_COLORS[model], pad=4)
            ax.grid(alpha=0.3)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

            if ri == 0 and ci == 1:
                ax.legend(fontsize=8, loc="lower right",
                          framealpha=0.95)

    fig.suptitle(
        "Pattern detection vs ablation detection — do the same heads win?",
        fontsize=14, fontweight="bold", y=1.0,
    )
    fig.text(
        0.5, -0.005,
        "Each dot = one of 720 heads. Top-10 by pattern (orange ring), top-10 by ablation (blue ring), "
        "intersection (filled, labeled). Strong agreement = many filled dots stacked top-right.",
        ha="center", fontsize=8.5, style="italic", color="#444",
    )
    fig.tight_layout()
    out_path = OUT_DIR / "pattern_vs_ablation_match.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_DIR / "pattern_vs_ablation_match.csv", index=False)
    print(f"saved {OUT_DIR / 'pattern_vs_ablation_match.csv'}")
    print()
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()

# %%
