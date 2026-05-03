# %%
"""Train TopK sparse autoencoders on the residual model's per-layer MLP delta
(branch_mlp output, BEFORE it is added back into the residual stream) at four
depths: L0, L9, L18, L35 (≈ 0%, 25%, 50%, ≈ 100% of model depth).

We hook `model.transformer.h[L].branch_mlp` because the user wants to study
"the delta the MLP wrote at each layer" — not the running residual sum.

TopK SAE (Anthropic / OpenAI style):
  encoder W_enc: D → k_dim     (k_dim = expansion * D)
  decoder W_dec: k_dim → D     (decoder columns unit-normed each step)
  z = ReLU(W_enc(x - b_pre) + b_enc)
  TopK: keep only the k largest components of z, zero the rest
  x_hat = W_dec(z) + b_pre

Loss = MSE(x_hat, x) (TopK provides sparsity for free; no L1 needed).

Outputs (per layer L):
  results/sae/residual_L{L:02d}/checkpoint.pt   # SAE state_dict
  results/sae/residual_L{L:02d}/training_log.csv  # iter, mse, l0, dead_pct, ev
  results/sae/residual_L{L:02d}/stats.json
  results/sae/training_curves.png                 # combined viz across layers

Run:
    uv run python src/mhc_interp/sae_train.py
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mhc_interp._loader import get_token_corpus, load_model_from_repo

device = "cuda" if torch.cuda.is_available() else "cpu"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = REPO_ROOT / "results" / "sae"

RESIDUAL_REPO = "Realmbird/mhc-781m-residual"
SAE_LAYERS = [0, 9, 18, 35]   # 0%, 25%, 50%, ≈100% of L=36

# Activation collection
N_TOKENS = 200_000           # SAEs need more data than linear AEs; 200K is a workable proof
SEQ_LEN = 512

# SAE architecture
EXPANSION = 8                # SAE_dim = 8 * D = 10240
TOPK_K = 64                  # at most 64/10240 = 0.6% active per token

# Training
BATCH_SIZE = 4096
N_ITERS = 5000               # ~5-10 min per layer on H100
LR = 5e-4
LR_WARMUP = 200
LR_DECAY_FRAC = 0.8          # cosine decay over the last 80% of training
LOG_EVERY = 100
DEAD_FEATURE_WINDOW = 1000   # iters of no activation → "dead"


# ---------- TopK SAE ----------
class TopKSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        # Tied init: decoder ≈ encoder transpose, then unit-normalize cols.
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    def encode(self, x):
        z = F.relu((x - self.b_pre) @ self.W_enc + self.b_enc)
        # TopK: keep top-k entries per token, zero the rest
        topk_vals, topk_idx = z.topk(self.k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_idx, topk_vals)
        return z_sparse

    def decode(self, z):
        return z @ self.W_dec + self.b_pre

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def normalize_decoder_(sae: TopKSAE):
    with torch.no_grad():
        sae.W_dec.div_(sae.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))


# ---------- activation collection ----------
@torch.no_grad()
def collect_branch_mlp_activations(model, ids: torch.Tensor, layer_idxs: list[int]) -> dict[int, torch.Tensor]:
    """Run forward over `ids` and capture branch_mlp output at each requested layer.
    Returns {layer: (N, D) fp32 tensor on CPU}."""
    captured: dict[int, list[torch.Tensor]] = {li: [] for li in layer_idxs}

    def make_hook(li):
        def hook(_m, _inp, out):
            captured[li].append(out.detach().cpu().to(torch.float32))
        return hook

    handles = []
    for li in layer_idxs:
        handles.append(model.transformer.h[li].branch_mlp.register_forward_hook(make_hook(li)))

    try:
        for batch in ids:
            x = batch.unsqueeze(0).to(device)
            _ = model(x)
    finally:
        for h in handles:
            h.remove()

    out = {}
    for li in layer_idxs:
        # captured[li] = [(1, T, D), ...]; concat to (N, D)
        cat = torch.cat([t.squeeze(0) for t in captured[li]], dim=0)
        out[li] = cat
    return out


# ---------- training loop ----------
def cosine_lr(it: int) -> float:
    if it < LR_WARMUP:
        return LR * it / max(1, LR_WARMUP)
    decay_start = int(N_ITERS * (1 - LR_DECAY_FRAC))
    if it < decay_start:
        return LR
    progress = (it - decay_start) / max(1, N_ITERS - decay_start)
    return LR * 0.5 * (1.0 + np.cos(np.pi * progress))


def train_sae(activations: torch.Tensor, layer_i: int) -> tuple[TopKSAE, list[dict], dict]:
    """activations: (N, D) on CPU. Returns trained SAE, training log, final stats."""
    N, D = activations.shape
    d_sae = EXPANSION * D
    sae = TopKSAE(D, d_sae, TOPK_K).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.999))

    log = []
    last_active_iter = torch.zeros(d_sae, dtype=torch.long, device=device)
    activations_gpu = activations.to(device)

    # Initialize b_pre to data mean (stabilizes training).
    with torch.no_grad():
        sae.b_pre.copy_(activations_gpu.mean(dim=0))

    rng = torch.Generator(device=device).manual_seed(42 + layer_i)

    t0 = time.time()
    for it in range(N_ITERS):
        # LR schedule
        for pg in opt.param_groups:
            pg["lr"] = cosine_lr(it)
        # Sample a batch
        idx = torch.randint(0, N, (BATCH_SIZE,), generator=rng, device=device)
        x = activations_gpu[idx]
        # Forward
        x_hat, z = sae(x)
        loss = F.mse_loss(x_hat, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Renormalize decoder rows after each step
        normalize_decoder_(sae)
        # Track which features fired this batch
        active_mask = (z.detach().abs() > 1e-9).any(dim=0)
        last_active_iter[active_mask] = it

        if (it + 1) % LOG_EVERY == 0 or it == 0:
            with torch.no_grad():
                # Held-out chunk = last 10% of activations (reproducible)
                held = activations_gpu[-min(10000, N // 10):]
                xh, zh = sae(held)
                mse = F.mse_loss(xh, held).item()
                # Explained variance computed on held-out: 1 - SS_res / SS_tot.
                # SS_tot uses the held set's own mean (so EV is comparable across layers
                # whose activation magnitudes differ by orders of magnitude).
                ss_res = ((held - xh) ** 2).sum().item()
                ss_tot = ((held - held.mean(dim=0, keepdim=True)) ** 2).sum().item()
                ev = 1.0 - ss_res / max(ss_tot, 1e-12)
                # L0 = avg # active features per token
                l0 = (zh.abs() > 1e-9).float().sum(dim=-1).mean().item()
                # Dead features: not active for DEAD_FEATURE_WINDOW iters
                dead = float((it - last_active_iter > DEAD_FEATURE_WINDOW).float().mean().item())
            log.append({"iter": it + 1, "lr": cosine_lr(it),
                        "train_mse": float(loss.item()), "held_mse": mse,
                        "l0": l0, "dead_pct": dead, "explained_variance": ev})

    elapsed = time.time() - t0
    final = log[-1].copy()
    final.update({"elapsed_sec": elapsed, "n_tokens": N, "d_in": D, "d_sae": d_sae,
                  "topk_k": TOPK_K})
    return sae, log, final


def run_one_layer(model, ids, layer_i: int, activations_cache: dict[int, torch.Tensor]):
    print(f"\n--- SAE @ residual L{layer_i} ---")
    if layer_i not in activations_cache:
        return  # caller should have populated
    acts = activations_cache[layer_i]
    print(f"  activations: {tuple(acts.shape)}")
    sae, log, final = train_sae(acts, layer_i)
    out_dir = RESULTS_ROOT / f"residual_L{layer_i:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": sae.state_dict(),
        "config": {"d_in": sae.d_in, "d_sae": sae.d_sae, "k": sae.k,
                   "layer": layer_i, "model": RESIDUAL_REPO,
                   "n_tokens": acts.shape[0]},
    }, out_dir / "checkpoint.pt")
    with open(out_dir / "training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader(); w.writerows(log)
    with open(out_dir / "stats.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"  done. mse={final['held_mse']:.4f}  L0={final['l0']:.1f}  "
          f"dead={final['dead_pct']*100:.1f}%  ev={final['explained_variance']:.3f}  "
          f"({final['elapsed_sec']:.0f}s)")


def make_combined_figure():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    metrics = [("held_mse", "Held-out MSE", axes[0, 0]),
               ("l0", "L0 (avg active features)", axes[0, 1]),
               ("dead_pct", "Dead-feature fraction", axes[1, 0]),
               ("explained_variance", "Explained variance", axes[1, 1])]
    cmap = plt.cm.viridis
    layer_names = sorted(p.name for p in RESULTS_ROOT.iterdir()
                         if p.is_dir() and p.name.startswith("residual_L"))
    for i, name in enumerate(layer_names):
        log_path = RESULTS_ROOT / name / "training_log.csv"
        if not log_path.exists():
            continue
        log = list(csv.DictReader(open(log_path)))
        x = [int(r["iter"]) for r in log]
        color = cmap(i / max(1, len(layer_names) - 1))
        for key, title, ax in metrics:
            ax.plot(x, [float(r[key]) for r in log], color=color, label=name.replace("residual_", ""))
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("iter")
            ax.grid(alpha=0.3)
    for _, _, ax in metrics:
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("TopK SAE training on residual model branch_mlp output", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig_path = RESULTS_ROOT / "training_curves.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_path}")


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"loading corpus: {N_TOKENS} tokens (Dolma/Pile) ...")
    ids = get_token_corpus(N_TOKENS, seq_len=SEQ_LEN, source="dolma")
    print(f"  corpus shape: {ids.shape}")

    print(f"\nloading {RESIDUAL_REPO} ...")
    model, cfg = load_model_from_repo(RESIDUAL_REPO, device)
    print(f"  L={cfg.n_layer}  D={cfg.n_embd}")

    print(f"capturing branch_mlp activations at layers {SAE_LAYERS} ...")
    activations = collect_branch_mlp_activations(model, ids, SAE_LAYERS)
    for li, acts in activations.items():
        print(f"  L{li}: {tuple(acts.shape)}  mean={acts.mean():.4f}  std={acts.std():.4f}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    for li in SAE_LAYERS:
        run_one_layer(None, ids, li, activations)

    make_combined_figure()


if __name__ == "__main__":
    main()

# %%
