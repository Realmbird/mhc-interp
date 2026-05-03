# %%
"""Logit-lens analysis averaged over a Dolma corpus.

The single-prompt logit-lens is noisy — at every layer the top-1 token can
flip arbitrarily because the residual stream encodes many things at once.
What's actually useful is the MEAN behavior across many natural prompts:

  * KL-to-final[L]  — how far is the lens at layer L from the model's final
                      output distribution? Drops as predictions sharpen.
  * top-1 acc[L]    — how often does the lens at layer L already pick the true
                      next token? Rises with depth.
  * mean logP_true  — log probability the lens at layer L assigns to the true
                      next token. Less ceiling-bound than top-1.
  * entropy[L]      — Shannon entropy of the lens distribution.

We compute all four per layer per model on N = 50,000 next-token-prediction
positions pulled from Dolma v1_7 (the variants' training distribution).

For mhc / mhc-lite the residual stream after each block is (B*S, T, D); we
sum over the S=4 streams to get the same (B, T, D) shape that ultimately
feeds the unembed (matches the model's reduce_stream).

Outputs (under `results/logit_lens/aggregate/`):
  metrics.csv         long form: model, layer, metric, value
  metrics.npz         per-(model, metric) (L,) arrays
  figure.png          4-panel: each metric vs layer with 3 model curves
  figure_diff.png     differences (mhc / mhc-lite minus residual)
  config.json         the run's settings + token count

Run:
    uv run python src/mhc_interp/logit_lens_mean.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from mhc_interp._loader import get_token_corpus, load_model_from_repo

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
N_TOKENS = 50_000      # ~390 seqs of 128 → ~49,610 prediction positions
SEQ_LEN = 128
BATCH_SEQS = 4         # forward pass at a time
METRICS = ["kl_to_final", "top1_acc", "mean_logp_true", "entropy"]
METRIC_LABELS = {
    "kl_to_final": "KL(lens || final)  — nats",
    "top1_acc": "top-1 accuracy on next token",
    "mean_logp_true": "mean log-P( true next token )",
    "entropy": "entropy of lens distribution  — nats",
}


def capture_residuals(model, ids: torch.Tensor) -> list[torch.Tensor]:
    """Returns a list of L raw post-MLP residuals (B*S, T, D) — DO NOT sum
    streams here. The lens applies ln_f BEFORE reducing, matching the model's
    final-layer flow `lm_head(reduce(ln_f(x)))`."""
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for li, block in enumerate(model.transformer.h):
        def hook(_m, _inp, out, li=li):
            captures[li] = out.detach()
        handles.append(block.hc_mlp.register_forward_hook(hook))
    try:
        with torch.no_grad():
            _ = model(ids)
    finally:
        for h in handles:
            h.remove()
    return [captures[li] for li in range(len(model.transformer.h))]


def lens_logits(model, x: torch.Tensor, S: int) -> torch.Tensor:
    """Apply the model-faithful lens: ln_f per stream, sum streams, lm_head.

    For S=1 this is just `lm_head(ln_f(x))`. For S>1, the model's final layer
    does sum_over_S(ln_f(stream_i)) before lm_head — we replicate that.
    """
    x = model.transformer.ln_f(x)
    if S > 1:
        Bs, T, D = x.shape
        x = x.reshape(Bs // S, S, T, D).sum(dim=1)  # (B, T, D)
    return model.lm_head(x)


@torch.no_grad()
def run_one_model(name: str, repo_id: str, corpus: torch.Tensor) -> dict[str, np.ndarray]:
    print(f"\n>>> {name}  ({repo_id})")
    model, cfg = load_model_from_repo(repo_id, device)
    L, S = cfg.n_layer, cfg.hyper_conn_n
    print(f"    L={L}  S={S}  type={cfg.hyper_conn_type}")

    sums = {m: np.zeros(L, dtype=np.float64) for m in METRICS}
    n_total = 0

    for batch_start in range(0, corpus.shape[0], BATCH_SEQS):
        ids = corpus[batch_start:batch_start + BATCH_SEQS].to(device)
        targets = ids[:, 1:]  # (B, T-1) — next-token targets

        residuals = capture_residuals(model, ids)  # raw (B*S, T, D) per layer

        # Reference final distribution = lens of last block's residual. With
        # the corrected lens (ln_f → reduce → lm_head), this matches the
        # model's actual output distribution.
        final_log_p = F.log_softmax(lens_logits(model, residuals[L - 1], S), dim=-1)
        final_p = final_log_p.exp()
        final_log_p_pred = final_log_p[:, :-1, :]
        final_p_pred = final_p[:, :-1, :]

        for li in range(L):
            x = residuals[li]
            log_p = F.log_softmax(lens_logits(model, x, S), dim=-1)
            log_p_pred = log_p[:, :-1, :]
            preds = log_p_pred.argmax(dim=-1)

            sums["top1_acc"][li] += (preds == targets).sum().item()
            sums["mean_logp_true"][li] += log_p_pred.gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1).sum().item()
            p = log_p_pred.exp()
            sums["entropy"][li] += -(p * log_p_pred).sum(dim=-1).sum().item()
            # KL(final || lens) — using final as the "true" target distribution
            # (forward KL = how much information you'd lose using lens as a
            # surrogate for final). Standard logit-lens convention.
            kl = (final_p_pred * (final_log_p_pred - log_p_pred)).sum(dim=-1)
            sums["kl_to_final"][li] += kl.sum().item()

            del log_p, log_p_pred, p
        del residuals, final_log_p, final_p, final_log_p_pred, final_p_pred

        n_total += int(targets.numel())

    means = {m: sums[m] / n_total for m in METRICS}

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return means


def plot_main(per_model: dict[str, dict[str, np.ndarray]], L: int):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, metric in zip(axes.flatten(), METRICS):
        for name, mets in per_model.items():
            ax.plot(
                np.arange(L), mets[metric],
                marker="o", markersize=3.5,
                color=MODEL_COLORS[name], label=name, linewidth=2,
            )
        ax.set_xlabel("layer")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(metric)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(
        f"Logit-lens metrics averaged over {N_TOKENS:,} Dolma next-token positions",
        fontsize=14, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    out = OUT_DIR / "figure.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_diff(per_model: dict[str, dict[str, np.ndarray]], L: int):
    """Differences mhc - residual and mhc_lite - residual, per metric."""
    base = per_model["residual"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, metric in zip(axes.flatten(), METRICS):
        for name in ["mhc", "mhc_lite"]:
            diff = per_model[name][metric] - base[metric]
            ax.plot(
                np.arange(L), diff,
                marker="o", markersize=3.5,
                color=MODEL_COLORS[name], label=f"{name} − residual",
                linewidth=2,
            )
        ax.axhline(0, color="black", lw=0.8, linestyle=":")
        ax.set_xlabel("layer")
        ax.set_ylabel(f"Δ {metric}")
        ax.set_title(f"{metric}: variant minus residual")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(
        "Per-layer logit-lens differences vs. residual baseline",
        fontsize=14, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    out = OUT_DIR / "figure_diff.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Pulling {N_TOKENS} tokens from {CORPUS_SOURCE}, seq_len={SEQ_LEN} ...")
    corpus = get_token_corpus(N_TOKENS, seq_len=SEQ_LEN, source=CORPUS_SOURCE)
    print(f"  → corpus shape: {tuple(corpus.shape)}  ({corpus.shape[0]} seqs)")

    per_model: dict[str, dict[str, np.ndarray]] = {}
    L = None
    for spec in MODELS:
        means = run_one_model(spec["name"], spec["repo_id"], corpus)
        per_model[spec["name"]] = means
        L = len(means[METRICS[0]])

    # ---- write CSV + NPZ
    csv_path = OUT_DIR / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "layer", "metric", "value"])
        for name, mets in per_model.items():
            for metric in METRICS:
                for li, v in enumerate(mets[metric]):
                    w.writerow([name, li, metric, f"{float(v):.6f}"])
    print(f"saved {csv_path}")

    npz_payload = {f"{name}__{metric}": per_model[name][metric]
                   for name in per_model for metric in METRICS}
    np.savez(OUT_DIR / "metrics.npz", **npz_payload)
    print(f"saved {OUT_DIR / 'metrics.npz'}")

    (OUT_DIR / "config.json").write_text(json.dumps({
        "corpus_source": CORPUS_SOURCE,
        "n_tokens": N_TOKENS,
        "seq_len": SEQ_LEN,
        "batch_seqs": BATCH_SEQS,
        "n_seqs": int(corpus.shape[0]),
        "n_prediction_positions": int(corpus.shape[0] * (corpus.shape[1] - 1)),
        "metrics": METRICS,
        "models": [s["name"] for s in MODELS],
    }, indent=2))

    # ---- figures
    plot_main(per_model, L)
    plot_diff(per_model, L)

    # ---- short summary print
    print("\n=== layer at which each metric crosses key thresholds ===")
    for name, mets in per_model.items():
        L_arr = np.arange(L)
        # earliest layer where top-1 acc reaches 90% of its final value
        top1 = mets["top1_acc"]
        target = top1[-1] * 0.9
        early = int(L_arr[top1 >= target][0]) if (top1 >= target).any() else -1
        # earliest layer where KL drops below 0.5
        kl = mets["kl_to_final"]
        kl_layer = int(L_arr[kl < 0.5][0]) if (kl < 0.5).any() else -1
        print(
            f"  {name:8s}  90% top-1 by L{early}  ·  KL<0.5 by L{kl_layer}  "
            f"·  final top-1={top1[-1]:.3f}  ·  final entropy={mets['entropy'][-1]:.2f}"
        )


if __name__ == "__main__":
    main()

# %%
