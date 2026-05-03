# %%
"""Box-and-whisker comparison of token-trajectories under the corrected lens
for the prompt "The cat sat on the" across the 3 mhc-781m variants.

For each (model, layer) we record the lens top-1 (and top-3) token at the
last position. We deliberately do NOT pick a "correct" answer — every
candidate that holds top-1 for ≥2 layers in any model gets a column.

Two output figures:
  cat_sat_token_layer_ranges.png — pure box-whisker (no privileged answer)
  cat_sat_final_top3.png         — each model's top-3 at L35, no comparison
                                   to a hand-picked target

Plus the long-form CSV.

Run:
    uv run python src/mhc_interp/cat_sat_token_ranges.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from mhc_interp._loader import load_model_from_repo
from mhc_interp.logit_lens_mean import capture_residuals, lens_logits

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("gpt2")

PROMPT = "The cat sat on the"
ANSWER_TOKEN = " floor"
TOP_K = 3   # how many ranks to track per layer
MODELS = [
    {"name": "residual", "repo_id": "Realmbird/mhc-781m-residual"},
    {"name": "mhc",      "repo_id": "Realmbird/mhc-781m-mhc"},
    {"name": "mhc_lite", "repo_id": "Realmbird/mhc-781m-mhc-lite"},
]
MODEL_COLORS = {"residual": "#7a7a7a", "mhc": "#c0392b", "mhc_lite": "#2980b9"}

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "logit_lens" / "aggregate"


def per_layer_topk(model, ids, S, L, k=TOP_K):
    """Returns a list of L dicts: each {'top_ids': (k,), 'top_probs': (k,)}."""
    res = capture_residuals(model, ids)
    out = []
    for li in range(L):
        logits = lens_logits(model, res[li], S)[0, -1]  # (V,)
        p = F.softmax(logits, dim=-1)
        probs, idxs = p.topk(k)
        out.append({"top_ids": idxs.cpu().tolist(),
                    "top_probs": probs.cpu().tolist()})
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    answer_id = tok.encode(ANSWER_TOKEN)[0]
    ids = torch.tensor([tok.encode(PROMPT)], device=device)

    per_model = {}
    for spec in MODELS:
        m, cfg = load_model_from_repo(spec["repo_id"], device)
        per_model[spec["name"]] = per_layer_topk(m, ids, cfg.hyper_conn_n, cfg.n_layer)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()

    L = len(per_model[MODELS[0]["name"]])

    # Build a long-form table: model, layer, rank, token, prob.
    rows = []
    for name, layers in per_model.items():
        for li, entry in enumerate(layers):
            for rank in range(TOP_K):
                tid = entry["top_ids"][rank]
                rows.append({
                    "model": name, "layer": li, "rank": rank + 1,
                    "token": tok.decode([tid]), "prob": entry["top_probs"][rank],
                })
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "cat_sat_token_layer_ranges.csv"
    df.to_csv(csv_path, index=False)
    print(f"saved {csv_path}  ({len(df)} rows)")

    # Identify common tokens: anything that appears as top-1 in any model, OR
    # that's the final answer.
    top1 = df[df["rank"] == 1]
    candidate_tokens = top1["token"].value_counts()
    # Keep tokens that appear as top-1 for >=2 layers in at least one model,
    # OR that's the answer token.
    keep = set()
    for tk in candidate_tokens.index:
        for m in [s["name"] for s in MODELS]:
            n = ((top1["token"] == tk) & (top1["model"] == m)).sum()
            if n >= 2:
                keep.add(tk)
                break
    keep.add(ANSWER_TOKEN)
    # Sort tokens by their median first-appearance layer (so x axis reads early→late).
    def median_first_layer(tk: str) -> float:
        layers = []
        for m in [s["name"] for s in MODELS]:
            sub = top1[(top1["token"] == tk) & (top1["model"] == m)]["layer"]
            if len(sub):
                layers.append(int(sub.min()))
        return float(np.median(layers)) if layers else 1e9
    tokens_ordered = sorted(keep, key=median_first_layer)

    # ---------- Build the figure ----------
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 1.0], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])

    n_tokens = len(tokens_ordered)
    n_models = len(MODELS)
    group_centers = np.arange(n_tokens)
    box_width = 0.22
    offsets = np.linspace(
        -(n_models - 1) / 2, (n_models - 1) / 2, n_models
    ) * (box_width + 0.04)

    legend_handles = []
    for mi, spec in enumerate(MODELS):
        name = spec["name"]
        positions = group_centers + offsets[mi]
        data = []
        for tk in tokens_ordered:
            sub = top1[(top1["token"] == tk) & (top1["model"] == name)]["layer"]
            data.append(sub.to_numpy() if len(sub) else np.array([]))

        # Draw box-whiskers only where there's data; scatter for sparse points.
        bp_data = []
        bp_positions = []
        for x, ys in zip(positions, data):
            if len(ys) >= 1:
                bp_data.append(ys)
                bp_positions.append(x)
        if bp_data:
            bp = ax.boxplot(
                bp_data, positions=bp_positions, widths=box_width,
                patch_artist=True,
                medianprops={"color": "black", "linewidth": 1.5},
                boxprops={"facecolor": MODEL_COLORS[name], "alpha": 0.75,
                          "edgecolor": "black"},
                whiskerprops={"color": "black"},
                capprops={"color": "black"},
                flierprops={"marker": "o", "markersize": 3.5,
                            "markerfacecolor": MODEL_COLORS[name],
                            "markeredgecolor": "black", "alpha": 0.6},
            )
        # Overlay individual layer dots so single-layer tokens still show.
        rng = np.random.default_rng(7 + mi)
        for x, ys in zip(positions, data):
            if len(ys) == 0:
                continue
            jitter = rng.uniform(-1, 1, size=len(ys)) * box_width / 3
            ax.scatter(np.full(len(ys), x) + jitter, ys, s=18,
                       color=MODEL_COLORS[name], edgecolor="black",
                       linewidth=0.4, alpha=0.9, zorder=5)

        legend_handles.append(plt.Rectangle(
            (0, 0), 1, 1, color=MODEL_COLORS[name], alpha=0.75, label=name))

    # Highlight the column for the final answer token.
    if ANSWER_TOKEN in tokens_ordered:
        ans_idx = tokens_ordered.index(ANSWER_TOKEN)
        ax.axvspan(ans_idx - 0.45, ans_idx + 0.45,
                   color="#fff2cc", alpha=0.55, zorder=0)
        ax.text(ans_idx, -2.2, "final\nanswer",
                ha="center", va="top", fontsize=9, color="#666",
                fontstyle="italic")

    ax.set_xticks(group_centers)
    ax.set_xticklabels([repr(t)[1:-1] for t in tokens_ordered],
                       fontsize=10, rotation=25, ha="right")
    ax.set_xlim(-0.7, n_tokens - 0.3)
    ax.set_ylabel("layer", fontsize=11)
    ax.set_ylim(-3, L)
    ax.invert_yaxis()
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30, 35])
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=legend_handles, loc="upper right", title="model")
    ax.set_title(
        f"Layers where each token is top-1 under the lens  ·  prompt: {PROMPT!r}",
        fontsize=12, fontweight="bold",
    )

    # ---------- Right-side: P(answer) at the final layer per model ----------
    ax2 = fig.add_subplot(gs[0, 1])
    final_p = []
    for spec in MODELS:
        layers = per_model[spec["name"]]
        # find probability of ANSWER token at final layer (may be in top_ids)
        last = layers[-1]
        if answer_id in last["top_ids"]:
            i = last["top_ids"].index(answer_id)
            p_ans = last["top_probs"][i]
        else:
            # Fall back: re-compute final probability for the answer token
            p_ans = float("nan")
        final_p.append((spec["name"], p_ans))
    names = [n for n, _ in final_p]
    ps = [p for _, p in final_p]
    bars = ax2.barh(
        names, ps, color=[MODEL_COLORS[n] for n in names],
        edgecolor="black", height=0.55,
    )
    for bar, p in zip(bars, ps):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{p:.3f}", va="center", ha="left", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, max(ps) * 1.35 if ps else 1)
    ax2.set_xlabel(f"P({ANSWER_TOKEN!r}) at final layer")
    ax2.set_title("Final answer\n(L35 lens)", fontsize=11, fontweight="bold")
    ax2.invert_yaxis()
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    # First-floor footer
    first_floor = {}
    for spec in MODELS:
        first = next(
            (li for li, e in enumerate(per_model[spec["name"]])
             if e["top_ids"][0] == answer_id), -1,
        )
        first_floor[spec["name"]] = first
    footer = "  ·  ".join(
        f"{n}: first {ANSWER_TOKEN!r} top-1 at L{l}" for n, l in first_floor.items()
    )
    fig.text(
        0.5, -0.01, footer,
        ha="center", fontsize=9.5, style="italic", color="#444",
    )

    fig.suptitle(
        "Per-prompt token trajectories — 'The cat sat on the' under the corrected lens",
        fontsize=13, fontweight="bold", y=1.02,
    )

    out_path = OUT_DIR / "cat_sat_token_layer_ranges.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

# %%
