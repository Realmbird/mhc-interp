# %%
"""Tight, embed-ready figure for the LessWrong writeup.

Combines:
  * Top-1 attention thumbnail per (detector, model) — visual character
  * Population-level counts (# heads with canonical-stripe-mass above 0.3 / 0.5 / 0.8)
    out of all 720 heads — defends against "you're cherry-picking n=1".

Output:
  results/analysis/head_pattern_writeup.png

Run:
    uv run python src/mhc_interp/head_pattern_writeup_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HEADS_ROOT = REPO_ROOT / "results" / "heads"
OUT_DIR = REPO_ROOT / "results" / "analysis"

MODELS = ["residual", "mhc", "mhc_lite"]
MODEL_COLORS = {"residual": "#7a7a7a", "mhc": "#c0392b", "mhc_lite": "#2980b9"}
THRESHOLDS = [0.3, 0.5, 0.8]


def stripe_indices(slug: str, T: int) -> tuple[np.ndarray, np.ndarray]:
    if slug == "induction":
        n = (T - 1) // 2
        rows = np.arange(1 + n, 1 + 2 * n - 1)
        cols = np.arange(2, n + 1)
    elif slug == "prev_token":
        rows = np.arange(1, T)
        cols = np.arange(0, T - 1)
    else:
        raise ValueError(slug)
    return rows, cols


def stripe_mass_all_heads(A: np.ndarray, slug: str) -> np.ndarray:
    L, H, T, _ = A.shape
    rows, cols = stripe_indices(slug, T)
    return A[:, :, rows, cols].mean(axis=-1).astype(np.float32)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    DETECTORS = [
        ("induction", "Induction heads"),
        ("prev_token", "Previous-token heads"),
    ]

    # Load everything once.
    data = {}
    for slug, _ in DETECTORS:
        per_model = {}
        for m in MODELS:
            A = np.load(HEADS_ROOT / slug / m / "attention.npy").astype(np.float32)
            top = json.loads((HEADS_ROOT / slug / m / "top_heads.json").read_text())
            stripe = stripe_mass_all_heads(A, slug)
            per_model[m] = {"A": A, "top": top, "stripe_mass_all": stripe}
        data[slug] = per_model

    fig = plt.figure(figsize=(13.5, 8))
    # 2 rows (one per detector) × 4 cols (3 thumbnails + 1 bar chart)
    gs = GridSpec(
        2, 4, figure=fig,
        width_ratios=[1, 1, 1, 1.2],
        height_ratios=[1, 1],
        hspace=0.45, wspace=0.30,
    )

    for ri, (slug, det_label) in enumerate(DETECTORS):
        # Row banner — placed via fig.text since GridSpec[ri, :] would need
        # an empty axis row.
        per_model = data[slug]
        T = per_model[MODELS[0]]["A"].shape[-1]
        rows, cols = stripe_indices(slug, T)

        # ---- 3 thumbnails (one per model, top-1 head) ----
        for ci, model in enumerate(MODELS):
            ax = fig.add_subplot(gs[ri, ci])
            A = per_model[model]["A"]
            top1 = per_model[model]["top"][0]
            li, hi = top1["layer"], top1["head"]
            P = A[li, hi]
            ax.imshow(P, cmap="viridis", vmin=0, vmax=1, aspect="equal")
            ax.plot(cols, rows, color="white", lw=0.6, alpha=0.55,
                    linestyle="--")
            sm = float(per_model[model]["stripe_mass_all"][li, hi])
            ax.set_title(
                f"{model}\nL{li}h{hi}  ·  stripe={sm:.2f}",
                fontsize=11, fontweight="bold", pad=4,
                color=MODEL_COLORS[model],
            )
            ax.set_xticks([])
            ax.set_yticks([])

            if ci == 0:
                ax.set_ylabel(det_label, fontsize=12, fontweight="bold",
                              labelpad=14)

        # ---- Population-count grouped bar chart ----
        ax_bar = fig.add_subplot(gs[ri, 3])
        x = np.arange(len(THRESHOLDS))
        width = 0.27
        offsets = np.linspace(-1, 1, len(MODELS)) * width
        for mi, model in enumerate(MODELS):
            v = per_model[model]["stripe_mass_all"].flatten()
            counts = [int((v > t).sum()) for t in THRESHOLDS]
            bars = ax_bar.bar(x + offsets[mi], counts, width=width,
                              color=MODEL_COLORS[model], alpha=0.85,
                              edgecolor="black", linewidth=0.5)
            for xi, c in zip(x + offsets[mi], counts):
                if c > 0:
                    ax_bar.text(xi, c + max(counts) * 0.02, str(c),
                                ha="center", va="bottom", fontsize=8.5)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f"> {t}" for t in THRESHOLDS])
        ax_bar.set_xlabel("canonical-stripe mass threshold", fontsize=9)
        ax_bar.set_ylabel("# heads (of 720)", fontsize=9)
        ax_bar.set_title("Population: heads above threshold",
                         fontsize=11, fontweight="bold", pad=4)
        # legend on top row only
        if ri == 0:
            handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m],
                                     alpha=0.85, label=m) for m in MODELS]
            ax_bar.legend(handles=handles, fontsize=9, loc="upper right",
                          frameon=False)
        ax_bar.tick_params(axis="x", labelsize=9)
        ax_bar.tick_params(axis="y", labelsize=8)
        for s in ("top", "right"):
            ax_bar.spines[s].set_visible(False)

    fig.suptitle(
        "Do MHC's attention heads behave like residual's? — top-1 thumbnail + population evidence",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, -0.012,
        "Top-1 head per (detector, model) chosen by the combined pattern+ablation score "
        "(pattern only for residual). White dashed line = canonical stripe — where the head "
        "should be looking if it implements that head type. "
        "Right column: # heads (of 720) whose canonical-stripe mass exceeds 0.3 / 0.5 / 0.8.",
        ha="center", fontsize=8.5, style="italic", color="#444",
    )
    fig.tight_layout()
    out_path = OUT_DIR / "head_pattern_writeup.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

# %%
