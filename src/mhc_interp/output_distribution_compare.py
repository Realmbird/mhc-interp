# %%
"""Test whether the 3 mhc-781m variants reach the *same destination* by
comparing their final-layer output distributions on a Dolma corpus.

Same final-layer top-1 accuracy is a 1-bit summary; the full output
distribution is 50,304-dim. Two models with identical top-1 can still
disagree by huge KL on the rest of the distribution.

For each pair (A, B) ∈ {residual, mhc, mhc_lite} we compute, averaged over
all (sequence × position) next-token positions in the corpus:

  KL(P_A || P_B)         (forward — extra nats encoding A under B)
  KL(P_B || P_A)         (reverse)
  Jensen-Shannon dist.   (symmetric, bounded log 2 ≈ 0.69)
  Total variation        (∈ [0, 1] — fraction of mass to move)
  top-1 agreement rate   (P(argmax A == argmax B))

For interpretation context we also report each model's mean entropy.

Outputs (under `results/logit_lens/aggregate/`):
  output_distribution_compare.csv   — per-pair metrics
  output_distribution_compare.png   — bar chart per metric
  output_distribution_compare.json  — config

Run:
    uv run python src/mhc_interp/output_distribution_compare.py
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from mhc_interp._loader import get_token_corpus, load_model_from_repo
from mhc_interp.logit_lens_mean import capture_residuals, lens_logits

device = "cuda" if torch.cuda.is_available() else "cpu"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "results" / "logit_lens" / "aggregate"

MODELS = [
    {"name": "residual", "repo_id": "Realmbird/mhc-781m-residual"},
    {"name": "mhc",      "repo_id": "Realmbird/mhc-781m-mhc"},
    {"name": "mhc_lite", "repo_id": "Realmbird/mhc-781m-mhc-lite"},
]
MODEL_COLORS = {"residual": "#7a7a7a", "mhc": "#c0392b", "mhc_lite": "#2980b9"}

CORPUS_SOURCE = "dolma"
N_TOKENS = 50_000
SEQ_LEN = 128
BATCH_SEQS = 4


@torch.no_grad()
def per_position_log_p(model, ids: torch.Tensor, S: int) -> torch.Tensor:
    """Run forward, capture the last block's residual, apply the model-faithful
    lens, return log-softmax at every position. Shape (B, T, V) on CPU fp16."""
    res = capture_residuals(model, ids)
    log_p = F.log_softmax(lens_logits(model, res[-1], S), dim=-1)
    return log_p.to(torch.float16).cpu()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Pulling {N_TOKENS} tokens from {CORPUS_SOURCE}, seq_len={SEQ_LEN}...")
    corpus = get_token_corpus(N_TOKENS, seq_len=SEQ_LEN, source=CORPUS_SOURCE)
    n_seqs, T = corpus.shape
    print(f"  → corpus shape: {tuple(corpus.shape)}  ({n_seqs} seqs × {T} tokens)")

    # Capture per-model log-softmax for every position (B, T, V), all on CPU.
    all_log_p: dict[str, list[torch.Tensor]] = {m["name"]: [] for m in MODELS}
    per_model_entropy_sum = {m["name"]: 0.0 for m in MODELS}
    n_total_positions = 0

    for spec in MODELS:
        name, repo = spec["name"], spec["repo_id"]
        print(f"\n>>> {name}  ({repo})")
        model, cfg = load_model_from_repo(repo, device)
        S, L = cfg.hyper_conn_n, cfg.n_layer

        ent_sum = 0.0
        n_pos = 0
        for batch_start in range(0, n_seqs, BATCH_SEQS):
            ids = corpus[batch_start:batch_start + BATCH_SEQS].to(device)
            log_p = per_position_log_p(model, ids, S)  # (B, T, V) fp16 CPU
            all_log_p[name].append(log_p)
            # entropy: -sum p log p (in nats)
            p = log_p.float().exp()
            ent = -(p * log_p.float()).sum(dim=-1).sum().item()
            ent_sum += ent
            n_pos += int(log_p.shape[0] * log_p.shape[1])
        per_model_entropy_sum[name] = ent_sum / n_pos
        n_total_positions = n_pos  # same across models
        print(f"    mean entropy: {per_model_entropy_sum[name]:.3f} nats  ({n_pos} positions)")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Concatenate over batches.
    for k in all_log_p:
        all_log_p[k] = torch.cat(all_log_p[k], dim=0)  # (n_seqs, T, V) fp16 CPU
        print(f"  {k}: {tuple(all_log_p[k].shape)}")

    # Compute pairwise stats.
    pair_stats = {}
    pair_per_position_kl = {}  # for the histogram
    print("\nComputing pairwise distribution distances...")
    for name_a, name_b in combinations([m["name"] for m in MODELS], 2):
        lp_a = all_log_p[name_a]  # (n_seqs, T, V)
        lp_b = all_log_p[name_b]
        # Stream over sequences to keep memory reasonable.
        kl_ab_sum = 0.0
        kl_ba_sum = 0.0
        js_sum = 0.0
        tv_sum = 0.0
        agree_sum = 0
        per_pos_kl_chunks = []
        for s_idx in range(0, lp_a.shape[0], BATCH_SEQS):
            la = lp_a[s_idx:s_idx + BATCH_SEQS].float()
            lb = lp_b[s_idx:s_idx + BATCH_SEQS].float()
            pa = la.exp()
            pb = lb.exp()
            # forward KL(A||B) per position
            kl_ab = (pa * (la - lb)).sum(dim=-1)         # (B, T)
            kl_ba = (pb * (lb - la)).sum(dim=-1)
            # Jensen-Shannon distance (symmetric)
            pm = 0.5 * (pa + pb)
            lm = (pm + 1e-30).log()
            js = 0.5 * (pa * (la - lm)).sum(dim=-1) + 0.5 * (pb * (lb - lm)).sum(dim=-1)
            # total variation
            tv = 0.5 * (pa - pb).abs().sum(dim=-1)        # (B, T) ∈ [0, 1]
            # top-1 agreement
            argmax_a = la.argmax(dim=-1)
            argmax_b = lb.argmax(dim=-1)
            agree = (argmax_a == argmax_b).sum().item()

            kl_ab_sum += kl_ab.sum().item()
            kl_ba_sum += kl_ba.sum().item()
            js_sum += js.sum().item()
            tv_sum += tv.sum().item()
            agree_sum += agree
            per_pos_kl_chunks.append(kl_ab.flatten().numpy())

        n = n_total_positions
        pair_stats[(name_a, name_b)] = {
            f"KL({name_a}||{name_b})_nats": kl_ab_sum / n,
            f"KL({name_b}||{name_a})_nats": kl_ba_sum / n,
            "JS_nats": js_sum / n,
            "TV": tv_sum / n,
            "top1_agreement": agree_sum / n,
        }
        pair_per_position_kl[(name_a, name_b)] = np.concatenate(per_pos_kl_chunks)
        print(f"  {name_a:8s} vs {name_b:8s}  "
              f"KL→={pair_stats[(name_a, name_b)][f'KL({name_a}||{name_b})_nats']:.3f}  "
              f"KL←={pair_stats[(name_a, name_b)][f'KL({name_b}||{name_a})_nats']:.3f}  "
              f"JS={pair_stats[(name_a, name_b)]['JS_nats']:.3f}  "
              f"TV={pair_stats[(name_a, name_b)]['TV']:.3f}  "
              f"top1-agree={pair_stats[(name_a, name_b)]['top1_agreement']:.3f}")

    # ---- Save CSV ----
    import csv
    csv_path = OUT_DIR / "output_distribution_compare.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "metric", "value"])
        for (a, b), st in pair_stats.items():
            for metric, val in st.items():
                w.writerow([f"{a}__{b}", metric, f"{val:.6f}"])
        for name, ent in per_model_entropy_sum.items():
            w.writerow([name, "mean_entropy_nats", f"{ent:.6f}"])
    print(f"saved {csv_path}")

    # ---- Figure ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    pair_labels = [f"{a}↔{b}" for a, b in pair_stats.keys()]

    # Pane 1: KL forward + reverse stacked side by side
    ax = axes[0, 0]
    pairs = list(pair_stats.keys())
    width = 0.38
    x = np.arange(len(pairs))
    kl_fwd = [pair_stats[p][f"KL({p[0]}||{p[1]})_nats"] for p in pairs]
    kl_rev = [pair_stats[p][f"KL({p[1]}||{p[0]})_nats"] for p in pairs]
    ax.bar(x - width / 2, kl_fwd, width, label="KL(A‖B)", color="#c0392b", alpha=0.85)
    ax.bar(x + width / 2, kl_rev, width, label="KL(B‖A)", color="#2980b9", alpha=0.85)
    for xi, v in zip(x - width / 2, kl_fwd):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + width / 2, kl_rev):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=10)
    ax.set_ylabel("nats")
    ax.set_title("KL divergence between final-layer distributions  (mean over 50k positions)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Pane 2: JS distance
    ax = axes[0, 1]
    js_vals = [pair_stats[p]["JS_nats"] for p in pairs]
    bars = ax.bar(x, js_vals, color="#7a548b", alpha=0.85)
    for xi, v in zip(x, js_vals):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(np.log(2), color="black", lw=0.8, linestyle=":", label=f"max = log 2 ≈ {np.log(2):.3f}")
    ax.set_xticks(x); ax.set_xticklabels(pair_labels)
    ax.set_ylabel("nats")
    ax.set_title("Jensen–Shannon divergence  (symmetric, bounded by log 2)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Pane 3: Total variation distance
    ax = axes[1, 0]
    tv_vals = [pair_stats[p]["TV"] for p in pairs]
    ax.bar(x, tv_vals, color="#27ae60", alpha=0.85)
    for xi, v in zip(x, tv_vals):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(pair_labels)
    ax.set_ylabel("TV  (fraction of mass to move)")
    ax.set_ylim(0, 1)
    ax.set_title("Total-variation distance  ∈ [0, 1]")
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Pane 4: top-1 agreement rate
    ax = axes[1, 1]
    agree_vals = [pair_stats[p]["top1_agreement"] for p in pairs]
    ax.bar(x, agree_vals, color="#f39c12", alpha=0.85)
    for xi, v in zip(x, agree_vals):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(pair_labels)
    ax.set_ylabel("P(argmax_A == argmax_B)")
    ax.set_ylim(0, 1)
    ax.set_title("Top-1 agreement rate  (the weak surrogate)")
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle(
        f"Do MHC variants reach the same destination?  "
        f"Per-pair output-distribution comparison  "
        f"({n_total_positions:,} Dolma positions)",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    out = OUT_DIR / "output_distribution_compare.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    (OUT_DIR / "output_distribution_compare.json").write_text(json.dumps({
        "n_positions": int(n_total_positions),
        "corpus": CORPUS_SOURCE,
        "seq_len": SEQ_LEN,
        "n_tokens_target": N_TOKENS,
        "models": [m["name"] for m in MODELS],
        "mean_entropy": per_model_entropy_sum,
        "pairs": {f"{a}__{b}": st for (a, b), st in pair_stats.items()},
    }, indent=2))


if __name__ == "__main__":
    main()

# %%
