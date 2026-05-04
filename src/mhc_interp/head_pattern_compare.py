# %%
"""Side-by-side attention-pattern comparison: do the top induction / prev-token
heads look identical across residual / mhc / mhc-lite, or does the architecture
twist the pattern?

For each detector we pick the model's top-1 head under the combined criterion
and render its attention on the same probe (synthetic [EOT]+R+R for induction;
natural sentence for prev-token). Annotated with the canonical stripe location
(induction = offset −n; prev-token = offset −1).

Output:
  results/analysis/head_pattern_compare.png

Run:
    uv run python src/mhc_interp/head_pattern_compare.py
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

MODELS = ["residual", "mhc", "mhc_lite"]
MODEL_COLORS = {"residual": "#7a7a7a", "mhc": "#c0392b", "mhc_lite": "#2980b9"}


def top1(slug: str, model: str) -> dict:
    return json.loads((HEADS_ROOT / slug / model / "top_heads.json").read_text())[0]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    detectors = [
        ("induction", "Induction heads (probe = [EOT]+R+R synthetic, n=25)"),
        ("prev_token", "Previous-token heads (probe = natural sentence)"),
    ]

    for row, (slug, row_title) in enumerate(detectors):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            top = top1(slug, model)
            li, hi = top["layer"], top["head"]
            A = np.load(HEADS_ROOT / slug / model / "attention.npy").astype(np.float32)
            tokens_meta = json.loads((HEADS_ROOT / slug / model / "tokens.json").read_text())
            tokens = tokens_meta["tokens"]
            T = A.shape[-1]
            P = A[li, hi]

            ax.imshow(P, cmap="viridis", vmin=0, vmax=1, aspect="equal")

            # Annotate the canonical stripe for that detector.
            if slug == "induction":
                # Probe layout: [EOT, R0..R_{n-1}, R0..R_{n-1}], length 1+2n.
                # Induction stripe: at row 1+n+i attend to col 1+i+1, for i in 0..n-2.
                n = (T - 1) // 2
                # Stripe is the diagonal offset by (n-1) below the row-anti-diag start.
                # Specifically rows = [1+n .. 2n-1], cols = [2 .. n].
                xs = np.arange(2, n + 1)
                ys = np.arange(1 + n, 2 * n)
                ax.plot(xs, ys, color="white", lw=0.7, alpha=0.55, linestyle="--")
                # Stripe-mass score:
                stripe = P[ys, xs] if len(xs) else np.array([])
                stripe_mass = stripe.mean() if len(stripe) else 0.0
                tick_positions = [0, 1, 1 + n, T - 1]
                tick_labels = ["EOT", "R₀", "R₀'", f"R{n - 1}'"]
            else:
                # prev-token stripe: A[i, i-1] for i in 1..T-1.
                xs = np.arange(0, T - 1)
                ys = np.arange(1, T)
                ax.plot(xs, ys, color="white", lw=0.7, alpha=0.55, linestyle="--")
                stripe = P[ys, xs]
                stripe_mass = stripe.mean()
                if T <= 14:
                    tick_positions = list(range(T))
                    tick_labels = [t.strip() or t for t in tokens]
                else:
                    tick_positions = list(range(0, T, max(1, T // 8)))
                    tick_labels = [(tokens[i].strip() or tokens[i])[:8] for i in tick_positions]

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=7)

            # Title block
            title = f"{model}  L{li}h{hi}  (canonical-stripe mass = {stripe_mass:.3f})"
            ax.set_title(title, fontsize=10, fontweight="bold",
                         color=MODEL_COLORS[model], pad=6)
            if col == 0:
                ax.set_ylabel("from (query)", fontsize=9)
            ax.set_xlabel("attended-to (key)", fontsize=9)

        # Row banner
        axes[row, 0].text(
            -0.3, 1.18, row_title,
            transform=axes[row, 0].transAxes,
            ha="left", va="bottom",
            fontsize=12, fontweight="bold",
        )

    fig.text(
        0.5, -0.01,
        "Dashed white line marks the canonical stripe for that detector. "
        "If the stripe lights up across all three models → the head is mechanistically the same.  "
        "Top-1 head per (detector, model) chosen by the combined pattern+ablation score "
        "(pattern only for residual; ablation skipped per user request).",
        ha="center", fontsize=8.5, style="italic", color="#444",
    )
    fig.suptitle(
        "Top-1 induction and prev-token heads — do the patterns match across architectures?",
        fontsize=13, fontweight="bold", y=1.005,
    )
    fig.tight_layout()
    out_path = OUT_DIR / "head_pattern_compare.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    # Quantitative table.
    rows = []
    for slug, _ in detectors:
        for model in MODELS:
            top = top1(slug, model)
            li, hi = top["layer"], top["head"]
            A = np.load(HEADS_ROOT / slug / model / "attention.npy").astype(np.float32)
            P = A[li, hi]
            T = P.shape[-1]
            if slug == "induction":
                n = (T - 1) // 2
                xs = np.arange(2, n + 1); ys = np.arange(1 + n, 2 * n)
            else:
                xs = np.arange(0, T - 1); ys = np.arange(1, T)
            stripe_mass = float(P[ys, xs].mean()) if len(xs) else 0.0
            mean_off_stripe = float((P.sum() - P[ys, xs].sum()) / max(P.size - len(xs), 1))
            rows.append({
                "detector": slug, "model": model, "layer": li, "head": hi,
                "stripe_mass": stripe_mass, "off_stripe_mean": mean_off_stripe,
                "stripe_to_off_ratio": stripe_mass / max(mean_off_stripe, 1e-9),
            })
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "head_pattern_compare.csv"
    df.to_csv(csv_path, index=False)
    print(f"saved {csv_path}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

# %%
