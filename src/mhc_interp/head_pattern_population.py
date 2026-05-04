# %%
"""Population-level pattern comparison: do top-K induction / prev-token heads
across the 3 mhc-781m variants exhibit the same canonical-stripe shape?

For each (detector, model):
  1. Top-5 heads (by combined score from top_heads.json) — attention thumbnails
     overlaid with the canonical stripe.
  2. Distribution of canonical-stripe-mass across ALL 720 heads, top-5 highlighted.

If the top-5 thumbnails light up the stripe consistently across all 3 models
AND the distributions have the same shape, the head TYPE is preserved at the
population level (not just for one cherry-picked head).

Output:
  results/analysis/head_pattern_population_induction.png
  results/analysis/head_pattern_population_prev_token.png
  results/analysis/head_pattern_population_stats.csv

Run:
    uv run python src/mhc_interp/head_pattern_population.py
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

MODELS = ["residual", "mhc", "mhc_lite"]
MODEL_COLORS = {"residual": "#7a7a7a", "mhc": "#c0392b", "mhc_lite": "#2980b9"}
TOP_K = 5
N_HEADS_TOTAL = 36 * 20  # 720


def stripe_indices(slug: str, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (rows, cols) of the canonical stripe for the detector."""
    if slug == "induction":
        n = (T - 1) // 2
        rows = np.arange(1 + n, 1 + 2 * n - 1)
        cols = np.arange(2, n + 1)
    elif slug == "prev_token":
        rows = np.arange(1, T)
        cols = np.arange(0, T - 1)
    else:
        raise ValueError(f"unsupported slug: {slug}")
    return rows, cols


def stripe_mass_all_heads(A: np.ndarray, slug: str) -> np.ndarray:
    """A: (L, H, T, T). Returns stripe_mass[L, H]."""
    L, H, T, _ = A.shape
    rows, cols = stripe_indices(slug, T)
    if len(rows) == 0:
        return np.zeros((L, H), dtype=np.float32)
    return A[:, :, rows, cols].mean(axis=-1).astype(np.float32)


def render_detector(slug: str, row_title: str, save_name: str):
    """One figure per detector: top-5 heads × 3 models × thumbnails + dist row."""
    # Load attention + tokens + top_heads per model.
    per_model = {}
    for m in MODELS:
        A = np.load(HEADS_ROOT / slug / m / "attention.npy").astype(np.float32)
        meta = json.loads((HEADS_ROOT / slug / m / "tokens.json").read_text())
        top = json.loads((HEADS_ROOT / slug / m / "top_heads.json").read_text())
        per_model[m] = {"A": A, "tokens": meta["tokens"], "top": top}

    T = per_model[MODELS[0]]["A"].shape[-1]
    rows, cols = stripe_indices(slug, T)

    fig = plt.figure(figsize=(13, 9))
    # 3 rows of thumbnails (one per model) × 5 cols + 1 row for distribution
    gs = GridSpec(
        4, TOP_K + 1, figure=fig,
        height_ratios=[1, 1, 1, 1.2],
        width_ratios=[1] * TOP_K + [0.05],
        hspace=0.45, wspace=0.18,
    )

    # ---- Top-K thumbnails per model ----
    pop_stats = {}
    for ri, model in enumerate(MODELS):
        A = per_model[model]["A"]
        tokens = per_model[model]["tokens"]
        top = per_model[model]["top"][:TOP_K]
        stripe = stripe_mass_all_heads(A, slug)  # (L, H)
        pop_stats[model] = stripe.flatten()

        for ci, entry in enumerate(top):
            ax = fig.add_subplot(gs[ri, ci])
            li, hi = entry["layer"], entry["head"]
            P = A[li, hi]
            ax.imshow(P, cmap="viridis", vmin=0, vmax=1, aspect="equal")
            ax.plot(cols, rows, color="white", lw=0.6, alpha=0.45,
                    linestyle="--")
            sm = stripe[li, hi]
            ax.set_title(f"L{li}h{hi}\nstripe={sm:.2f}",
                         fontsize=9, color=MODEL_COLORS[model],
                         fontweight="bold", pad=4)
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(model, fontsize=11, fontweight="bold",
                              color=MODEL_COLORS[model], labelpad=10)

    # ---- Distribution row: histogram over all 720 stripe-mass values ----
    bins = np.linspace(0, 1, 41)
    for ci, model in enumerate(MODELS):
        ax = fig.add_subplot(gs[3, ci])
        vals = pop_stats[model]
        ax.hist(vals, bins=bins, color=MODEL_COLORS[model], alpha=0.85,
                edgecolor="black", linewidth=0.4)
        # Mark top-5 stripe-mass values
        top_vals = sorted([float(stripe_mass_all_heads(per_model[model]["A"], slug)
                                   [e["layer"], e["head"]])
                            for e in per_model[model]["top"][:TOP_K]], reverse=True)
        for v in top_vals:
            ax.axvline(v, color="black", lw=1.2, alpha=0.65)
        median = float(np.median(vals))
        ax.axvline(median, color="black", lw=0.8, linestyle=":", alpha=0.6)

        n_above = (vals > 0.5).sum()
        ax.set_title(
            f"{model}: median={median:.3f}  ·  {n_above}/720 heads with stripe>0.5",
            fontsize=9.5, color=MODEL_COLORS[model],
        )
        ax.set_xlabel("canonical-stripe mass", fontsize=8)
        if ci == 0:
            ax.set_ylabel("# heads", fontsize=8)
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.text(
        0.5, 1.005,
        f"{row_title}",
        ha="center", fontsize=13, fontweight="bold",
    )
    fig.text(
        0.5, -0.005,
        "Top 3 rows: attention thumbnails for top-5 heads per model (combined criterion). "
        "Dashed white line = canonical stripe.  "
        "Bottom row: histogram of canonical-stripe mass across all 720 heads, with "
        "vertical lines = top-5 head values.  log y-axis.",
        ha="center", fontsize=8.5, style="italic", color="#444",
    )
    out_path = OUT_DIR / save_name
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return pop_stats


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pops = {}
    for slug, title, fname in [
        ("induction",
         "Induction heads — top-5 thumbnails + population distribution",
         "head_pattern_population_induction.png"),
        ("prev_token",
         "Previous-token heads — top-5 thumbnails + population distribution",
         "head_pattern_population_prev_token.png"),
    ]:
        pops[slug] = render_detector(slug, title, fname)

    # Per-model summary stats CSV: median, fraction above thresholds.
    rows = []
    for slug in pops:
        for model in MODELS:
            v = pops[slug][model]
            rows.append({
                "detector": slug,
                "model": model,
                "n_heads": len(v),
                "median_stripe_mass": float(np.median(v)),
                "mean_stripe_mass": float(v.mean()),
                "max_stripe_mass": float(v.max()),
                "n_above_0.3": int((v > 0.3).sum()),
                "n_above_0.5": int((v > 0.5).sum()),
                "n_above_0.8": int((v > 0.8).sum()),
            })
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "head_pattern_population_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"saved {csv_path}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

# %%
