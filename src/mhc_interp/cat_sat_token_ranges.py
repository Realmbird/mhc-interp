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


# Token-selection threshold for the box-whisker: a candidate token is shown
# if it holds top-1 (rank 1) for at least this many layers in at least one of
# the 3 models. ≥2 filters out single-layer flickers (likely noise) without
# privileging any specific "correct" answer.
MIN_TOP1_LAYERS = 2


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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

    # ---- Token selection (no privileged "answer"):
    # Keep any token that holds top-1 for >= MIN_TOP1_LAYERS in at least one model.
    top1 = df[df["rank"] == 1]
    keep = set()
    for tk in top1["token"].unique():
        for m in [s["name"] for s in MODELS]:
            n = ((top1["token"] == tk) & (top1["model"] == m)).sum()
            if n >= MIN_TOP1_LAYERS:
                keep.add(tk)
                break
    # Sort tokens by their median first-appearance layer (so x axis reads early→late).
    def median_first_layer(tk: str) -> float:
        layers = []
        for m in [s["name"] for s in MODELS]:
            sub = top1[(top1["token"] == tk) & (top1["model"] == m)]["layer"]
            if len(sub):
                layers.append(int(sub.min()))
        return float(np.median(layers)) if layers else 1e9
    tokens_ordered = sorted(keep, key=median_first_layer)

    # ---------- Figure 1: pure box-whisker, no privileged answer ----------
    fig, ax = plt.subplots(figsize=(12, 6.5))

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

    ax.set_xticks(group_centers)
    ax.set_xticklabels([repr(t)[1:-1] for t in tokens_ordered],
                       fontsize=10, rotation=25, ha="right")
    ax.set_xlim(-0.7, n_tokens - 0.3)
    ax.set_ylabel("layer", fontsize=11)
    ax.set_ylim(-1.5, L)
    ax.invert_yaxis()
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30, 35])
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=legend_handles, loc="upper right", title="model")
    ax.set_title(
        f"Layers where each token holds top-1 under the lens  ·  prompt: {PROMPT!r}",
        fontsize=12, fontweight="bold",
    )
    fig.text(
        0.5, -0.02,
        f"Token-selection rule: token shown iff it holds top-1 (rank 1) for ≥ {MIN_TOP1_LAYERS} "
        f"layers in at least one model.  No token is privileged as 'the answer'.",
        ha="center", fontsize=9, style="italic", color="#444",
    )
    fig.tight_layout()
    out_path = OUT_DIR / "cat_sat_token_layer_ranges.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    # ---------- Figure 2: each model's top-3 at the final layer ----------
    # Separate file. No comparison across models on a hand-picked target token —
    # we just show what each model actually predicts at L35.
    fig2, axes = plt.subplots(1, len(MODELS), figsize=(13, 4.0), sharex=False)
    for ax2, spec in zip(axes, MODELS):
        name = spec["name"]
        last = per_model[name][-1]
        toks = [tok.decode([i]) for i in last["top_ids"]]
        probs = list(last["top_probs"])
        bars = ax2.barh(
            range(len(toks)), probs,
            color=MODEL_COLORS[name], alpha=0.85, edgecolor="black",
        )
        for i, (b, p) in enumerate(zip(bars, probs)):
            ax2.text(b.get_width() + 0.003, b.get_y() + b.get_height() / 2,
                     f"{p:.3f}", va="center", ha="left", fontsize=10,
                     fontweight="bold")
        ax2.set_yticks(range(len(toks)))
        ax2.set_yticklabels([repr(t)[1:-1] for t in toks],
                            fontsize=11, fontfamily="monospace")
        ax2.invert_yaxis()
        ax2.set_xlim(0, max(probs) * 1.35)
        ax2.set_xlabel("probability")
        ax2.set_title(name, fontsize=12, fontweight="bold",
                      color=MODEL_COLORS[name])
        for s in ("top", "right"):
            ax2.spines[s].set_visible(False)
    fig2.suptitle(
        f"Top-{TOP_K} tokens at the final layer (L{L - 1}) — each model on its own terms",
        fontsize=12.5, fontweight="bold", y=1.02,
    )
    fig2.text(
        0.5, -0.04,
        f"Prompt: {PROMPT!r}.  No 'correct answer' selected — these are simply the "
        f"top-{TOP_K} most-likely next tokens under each model's actual output distribution.",
        ha="center", fontsize=9, style="italic", color="#444",
    )
    fig2.tight_layout()
    out2 = OUT_DIR / "cat_sat_final_top3.png"
    fig2.savefig(out2, dpi=170, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()

# %%
