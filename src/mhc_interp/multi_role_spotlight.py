# %%
"""Spotlight figure showing multi-role heads in BOTH residual and MHC, since
the user (rightly) pointed out that multi-role heads aren't unique to MHC —
the differential is at 3-role concentration.

Top row: residual's three strongest 2-role heads.
Bottom row: MHC's spotlight — the strongest 2-role + the two 3-role heads.

Each panel: attention pattern (induction probe of that model) + bar chart of
the head's 5 detector scores, top-10 cells starred.

Output:
  results/analysis/multi_role/spotlight.png

Run:
    uv run python src/mhc_interp/multi_role_spotlight.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HEADS_ROOT = REPO_ROOT / "results" / "heads"
OUT_DIR = REPO_ROOT / "results" / "analysis" / "multi_role"

DETECTORS = ["prev_token", "induction", "duplicate", "successor", "copy_suppression"]
DET_LABELS = {
    "prev_token": "prev-token",
    "induction": "induction",
    "duplicate": "duplicate",
    "successor": "successor",
    "copy_suppression": "copy-suppr",
}

# Top-3 multi-role heads per row, picked from the verified CSV.
ROWS = [
    ("residual", [
        (15, 12, "prev-token + successor"),
        (16, 17, "prev-token + successor"),
        (5,  11, "duplicate + successor"),
    ]),
    ("mhc", [
        (7, 17, "induction + copy-suppr"),
        (7, 15, "duplicate + successor + copy-suppr"),
        (6, 11, "prev-token + successor + copy-suppr"),
    ]),
]

TOP_K = 10


def load_score_table(model: str) -> dict[str, pd.DataFrame]:
    return {d: pd.read_csv(HEADS_ROOT / d / model / "scores.csv") for d in DETECTORS}


def head_score(table: dict[str, pd.DataFrame], detector: str, layer: int, head: int) -> float:
    df = table[detector]
    return float(df[(df["layer"] == layer) & (df["head"] == head)]["score"].iloc[0])


def head_is_top_k(table: dict[str, pd.DataFrame], detector: str, layer: int, head: int, k: int = TOP_K) -> bool:
    df = table[detector].sort_values("score", ascending=False).reset_index(drop=True)
    rank = df.index[(df["layer"] == layer) & (df["head"] == head)][0]
    return int(rank) < k


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load attention NPYs for the induction probe of each model so we render
    # the same probe across both rows (synthetic [EOT] + R + R, n=25).
    attn = {model: np.load(HEADS_ROOT / "induction" / model / "attention.npy").astype(np.float32)
            for model in {row[0] for row in ROWS}}
    score_tables = {model: load_score_table(model) for model in {row[0] for row in ROWS}}

    fig = plt.figure(figsize=(14, 13))
    n_rows = len(ROWS)
    n_cols = max(len(spotlights) for _, spotlights in ROWS)
    # Two grid rows per spotlight row (attention thumb + score bars), plus a
    # spacer row before each spotlight row to make room for the banner.
    n_grid_rows = n_rows * 3  # banner | thumb | bars  per spotlight row
    height_ratios = []
    for _ in range(n_rows):
        height_ratios.extend([0.18, 1.55, 1.0])
    gs = GridSpec(
        n_grid_rows, n_cols, figure=fig,
        height_ratios=height_ratios,
        hspace=0.65, wspace=0.32,
    )

    for row_idx, (model, spotlights) in enumerate(ROWS):
        A = attn[model]
        T = A.shape[-1]
        n = (T - 1) // 2
        max_n_roles = max(
            sum(head_is_top_k(score_tables[model], d, l, h) for d in DETECTORS)
            for l, h, _ in spotlights
        )
        # Banner subplot — spans the whole width of the row.
        ax_banner = fig.add_subplot(gs[row_idx * 3, :])
        ax_banner.axis("off")
        ax_banner.text(
            0.0, 0.5,
            f"{model.upper()}  —  multi-role spotlight"
            f"  ·  this row's max concentration: {max_n_roles}-role",
            transform=ax_banner.transAxes, ha="left", va="center",
            fontsize=13, fontweight="bold",
            color="#7a7a7a" if model == "residual" else "#c0392b",
        )

        for col_idx, (li, hi, byline) in enumerate(spotlights):
            ax_att = fig.add_subplot(gs[row_idx * 3 + 1, col_idx])
            pattern = A[li, hi]
            ax_att.imshow(pattern, cmap="viridis", vmin=0, vmax=1, aspect="equal")
            n_roles = sum(head_is_top_k(score_tables[model], d, li, hi) for d in DETECTORS)
            ax_att.set_title(
                f"L{li}  h{hi}    ({n_roles}-role)",
                fontsize=11, fontweight="bold", pad=4,
            )
            ax_att.text(
                0.5, 1.07, byline,
                transform=ax_att.transAxes, ha="center", va="bottom",
                fontsize=9, style="italic", color="#555",
            )
            ax_att.set_xticks([0, 1, 1 + n, T - 1])
            ax_att.set_xticklabels(["EOT", "R₀", "R₀'", f"R{n - 1}'"], fontsize=7)
            ax_att.set_yticks([0, 1, 1 + n, T - 1])
            ax_att.set_yticklabels(["EOT", "R₀", "R₀'", f"R{n - 1}'"], fontsize=7)
            if col_idx == 0:
                ax_att.set_ylabel("from (query)", fontsize=8)
            ax_att.set_xlabel("attended-to (key)", fontsize=8)

            ax_bar = fig.add_subplot(gs[row_idx * 3 + 2, col_idx])
            scores = [head_score(score_tables[model], d, li, hi) for d in DETECTORS]
            top_flags = [head_is_top_k(score_tables[model], d, li, hi) for d in DETECTORS]
            colors = ["#c0392b" if t else "#bdc3c7" for t in top_flags]
            labels = [DET_LABELS[d] for d in DETECTORS]
            bars = ax_bar.barh(labels, scores, color=colors, edgecolor="black")
            for bar, v, top in zip(bars, scores, top_flags):
                ha = "left" if v >= 0 else "right"
                offset = 0.02 if v >= 0 else -0.02
                txt = f"{v:.2f}" if v >= 0 else f"{v:+.2f}"
                ax_bar.text(
                    v + offset, bar.get_y() + bar.get_height() / 2,
                    txt + ("  ★" if top else ""),
                    va="center", ha=ha, fontsize=8.5,
                    fontweight="bold" if top else "normal",
                )
            ax_bar.invert_yaxis()
            ax_bar.set_xlim(min(min(scores) - 0.10, -0.10), max(scores) + 0.30)
            ax_bar.axvline(0, color="black", lw=0.5)
            ax_bar.tick_params(axis="x", labelsize=8)
            ax_bar.set_xlabel("detector score", fontsize=8)
            for s in ("top", "right"):
                ax_bar.spines[s].set_visible(False)

    fig.text(
        0.5, 0.005,
        "★ = head ranks in top-10 of 720 (L=36 × H=20) for that detector. "
        "All attention thumbnails computed on the synthetic induction probe [EOT]+R+R, n=25. "
        "Residual/MHC totals: 6/7 multi-role heads (≥2 roles); 3-role heads appear only in MHC.",
        ha="center", fontsize=8.5, style="italic", color="#444",
    )

    fig.suptitle(
        "Multi-role attention heads in residual vs. MHC  —  similar prevalence, different concentration",
        fontsize=14, fontweight="bold", y=0.995,
    )

    out_path = OUT_DIR / "spotlight.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

# %%
