# %%
"""Multi-role head analysis across the 3 mhc-781m variants.

For each (model, layer, head) load the score under all 5 detectors, then look
for heads that score highly under more than one detector. Output a visual the
user can drop into a writeup, plus a CSV of the actual numbers — so claims
about "L7h17 is dual induction + copy-suppression" are verifiable, not vibes.

Outputs:
  results/analysis/multi_role/multi_role_heads.png   — main figure
  results/analysis/multi_role/role_count_distribution.png  — supporting bar chart
  results/analysis/multi_role/multi_role_heads.csv   — per-head rank matrix

Run:
    uv run python src/mhc_interp/multi_role_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

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
MODELS = ["residual", "mhc", "mhc_lite"]

# A head is considered to "play role X" if it ranks in the top-K heads for
# detector X in that model. K=10 out of 720 ≈ top ~1.4%.
TOP_K = 10
# Multi-role threshold: how many roles to qualify for the multi-role panel.
MULTI_ROLE_MIN = 2


def load_scores(model: str) -> dict[str, np.ndarray]:
    """Returns {detector: scores (L, H)}."""
    out = {}
    for det in DETECTORS:
        df = pd.read_csv(HEADS_ROOT / det / model / "scores.csv")
        L = int(df["layer"].max()) + 1
        H = int(df["head"].max()) + 1
        S = np.zeros((L, H), dtype=np.float32)
        S[df["layer"].to_numpy(), df["head"].to_numpy()] = df["score"].to_numpy()
        out[det] = S
    return out


def percentile_rank(M: np.ndarray) -> np.ndarray:
    """Return rank ∈ [0, 1] for every entry, 1.0 = top score."""
    flat = M.flatten()
    order = flat.argsort()
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(order), dtype=np.float32)
    ranks /= max(len(flat) - 1, 1)
    return ranks.reshape(M.shape)


def top_k_heads(scores: np.ndarray, k: int) -> set[tuple[int, int]]:
    """Return {(layer, head)} of the top-k entries by score."""
    L, H = scores.shape
    flat = [(li, hi, float(scores[li, hi])) for li in range(L) for hi in range(H)]
    flat.sort(key=lambda r: -r[2])
    return {(li, hi) for li, hi, _ in flat[:k]}


def per_head_role_table(model: str) -> pd.DataFrame:
    """One row per (layer, head). Columns: raw score and percentile per detector,
    role count under TOP_K, plus role membership flags."""
    scores = load_scores(model)
    L, H = scores[DETECTORS[0]].shape
    pct = {d: percentile_rank(s) for d, s in scores.items()}
    top_sets = {d: top_k_heads(s, TOP_K) for d, s in scores.items()}

    rows = []
    for li in range(L):
        for hi in range(H):
            row = {"model": model, "layer": li, "head": hi}
            roles = []
            for d in DETECTORS:
                row[f"{d}_score"] = float(scores[d][li, hi])
                row[f"{d}_pct"] = float(pct[d][li, hi])
                row[f"{d}_top{TOP_K}"] = int((li, hi) in top_sets[d])
                if (li, hi) in top_sets[d]:
                    roles.append(d)
            row["n_roles"] = len(roles)
            row["roles"] = "+".join(roles) if roles else ""
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_model = {m: per_head_role_table(m) for m in MODELS}
    full_df = pd.concat(per_model.values(), ignore_index=True)
    csv_path = OUT_DIR / "multi_role_heads.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"saved {csv_path}  ({len(full_df)} rows)")

    # =========================================================================
    # Figure 1: per-model heatmap of multi-role heads × detectors.
    # Rows: heads with n_roles >= MULTI_ROLE_MIN, sorted by (n_roles desc,
    # max_pct desc). Cell color: percentile rank; cell annotation: raw score.
    # =========================================================================
    fig, axes = plt.subplots(
        1, len(MODELS),
        figsize=(15, 8),
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )

    cmap = LinearSegmentedColormap.from_list(
        "white_to_red", ["#f7f7f7", "#fdcc8a", "#e34a33", "#7f0000"]
    )

    multi_role_summary = []
    for ax, model in zip(axes, MODELS):
        df = per_model[model]
        multi = df[df["n_roles"] >= MULTI_ROLE_MIN].copy()
        # sort by n_roles desc, then by max percentile across detectors desc
        max_pct = multi[[f"{d}_pct" for d in DETECTORS]].max(axis=1)
        multi = multi.assign(_max_pct=max_pct).sort_values(
            ["n_roles", "_max_pct"], ascending=[False, False]
        )
        # cap at 15 rows for readability
        multi = multi.head(15)

        labels = [f"L{r.layer}  h{r.head}" for r in multi.itertuples(index=False)]
        pct_matrix = multi[[f"{d}_pct" for d in DETECTORS]].to_numpy()
        score_matrix = multi[[f"{d}_score" for d in DETECTORS]].to_numpy()
        top_matrix = multi[[f"{d}_top{TOP_K}" for d in DETECTORS]].to_numpy()

        if pct_matrix.size == 0:
            ax.text(0.5, 0.5, f"{model}\n(no multi-role heads)",
                    ha="center", va="center", fontsize=12, transform=ax.transAxes)
            ax.set_axis_off()
            continue

        im = ax.imshow(pct_matrix, cmap=cmap, vmin=0.85, vmax=1.0, aspect="auto")
        for i in range(pct_matrix.shape[0]):
            for j in range(pct_matrix.shape[1]):
                txt = f"{score_matrix[i, j]:+.2f}" if DETECTORS[j] in ("successor", "copy_suppression") \
                    else f"{score_matrix[i, j]:.2f}"
                color = "white" if pct_matrix[i, j] > 0.97 else "black"
                weight = "bold" if top_matrix[i, j] else "normal"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color=color, fontweight=weight)
                if top_matrix[i, j]:
                    # mark top-K cells with a black border
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               fill=False, edgecolor="black",
                                               linewidth=1.5))

        ax.set_xticks(range(len(DETECTORS)))
        ax.set_xticklabels([DET_LABELS[d] for d in DETECTORS], rotation=30, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"{model}  ({(df['n_roles'] >= MULTI_ROLE_MIN).sum()} heads with ≥{MULTI_ROLE_MIN} roles)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("detector")

        for r in multi.itertuples(index=False):
            multi_role_summary.append({
                "model": model, "layer": r.layer, "head": r.head,
                "n_roles": r.n_roles, "roles": r.roles,
            })

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("percentile rank within detector (0.85 → 1.0)")
    fig.suptitle(
        f"Multi-role heads across the 3 mhc-781m variants  "
        f"(top-{TOP_K} of 720 in ≥{MULTI_ROLE_MIN} detectors; cells = raw score, bordered = top-{TOP_K})",
        fontsize=13, fontweight="bold"
    )
    fig_path = OUT_DIR / "multi_role_heads.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_path}")

    # =========================================================================
    # Figure 2: distribution — # heads with N roles, per model.
    # Verifies whether MHC variants concentrate roles into fewer heads.
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_roles_max = max(int(df["n_roles"].max()) for df in per_model.values())
    bar_x = np.arange(1, n_roles_max + 1)
    width = 0.27
    for offset, model in zip([-1, 0, 1], MODELS):
        df = per_model[model]
        counts = [(df["n_roles"] == n).sum() for n in bar_x]
        ax.bar(bar_x + offset * width, counts, width=width, label=model)
        for x, c in zip(bar_x + offset * width, counts):
            if c > 0:
                ax.text(x, c + 0.05, str(c), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel(f"# detectors a head is top-{TOP_K} in")
    ax.set_ylabel("# heads")
    ax.set_xticks(bar_x)
    ax.set_title(f"Multi-role count distribution  (top-{TOP_K} per detector, out of 720 heads/model)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig_path = OUT_DIR / "role_count_distribution.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_path}")

    # =========================================================================
    # Print verification text for the L7h17 / L7h15 claims.
    # =========================================================================
    print("\n" + "=" * 80)
    print("Verifying the multi-role claims")
    print("=" * 80)
    for model in MODELS:
        df = per_model[model]
        for li, hi in [(7, 17), (7, 15), (6, 11)]:
            row = df[(df["layer"] == li) & (df["head"] == hi)]
            if row.empty:
                continue
            r = row.iloc[0]
            roles = r["roles"] or "—"
            scores_s = " ".join(
                f"{DET_LABELS[d]}={r[f'{d}_score']:+.3f}(top{TOP_K}={int(r[f'{d}_top{TOP_K}'])})"
                for d in DETECTORS
            )
            print(f"  {model:8s}  L{li}h{hi}  n_roles={r['n_roles']}  [{roles}]")
            print(f"             {scores_s}")
        print()

    # Summary of multi-role counts:
    print("Multi-role head counts per model:")
    for model in MODELS:
        df = per_model[model]
        for n in range(1, n_roles_max + 1):
            c = int((df["n_roles"] == n).sum())
            if c:
                print(f"  {model:8s}  exactly {n} role(s): {c} heads")
        print()


if __name__ == "__main__":
    main()

# %%
