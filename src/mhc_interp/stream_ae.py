# %%
"""Train autoencoders between residual streams in MHC and MHC-lite.

For each (model ∈ {mhc, mhc_lite}, layer L, src stream i, dst stream j) we
fit two AEs that map activations of stream i at layer L → stream j at layer L:
  * a linear map (closed-form OLS, ridge-regularized)
  * a 1-hidden-layer MLP (D → D/2 → D, GELU, trained with Adam)

R² = 1 - residual_variance / total_variance is reported on a held-out 20%
split. Alongside, we extract per-layer norms of the hyper-conn parameters
(static_alpha, static_beta, mean ||dynamic_alpha||, mean ||dynamic_beta||)
so the user can compare AE accuracy against the actual mixing magnitudes.

Outputs (per model):
  results/stream_ae/{model}/r2_linear.npy      # (L, S, S) — diagonal = trivial 1.0
  results/stream_ae/{model}/r2_mlp.npy
  results/stream_ae/{model}/alpha_beta.csv     # one row per layer
  results/stream_ae/{model}/cross_stream_summary.csv
  results/stream_ae/r2_vs_beta.png             # combined viz

Run:
    uv run python src/mhc_interp/stream_ae.py
"""
from __future__ import annotations

import csv
import json
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mhc_interp._loader import get_token_corpus, load_model_from_repo

device = "cuda" if torch.cuda.is_available() else "cpu"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = REPO_ROOT / "results" / "stream_ae"

MODELS = [
    {"name": "mhc",      "repo_id": "Realmbird/mhc-781m-mhc"},
    {"name": "mhc_lite", "repo_id": "Realmbird/mhc-781m-mhc-lite"},
]

N_TOKENS = 50_000     # ≈ 100 sequences × 512 tokens — enough for a stable linear fit
SEQ_LEN = 512
RIDGE = 1e-2          # ridge λ for linear AE; small but stable for D=1280
MLP_EPOCHS = 4        # MLP AE training epochs
MLP_BATCH = 4096
MLP_LR = 3e-4


# ---------- collect activations ----------
@torch.no_grad()
def capture_post_mlp_streams(model, cfg, ids: torch.Tensor) -> torch.Tensor:
    """Run forward pass, hook each block's hc_mlp output (post-MLP residual
    streams), return (L, S, N, D) on CPU as float16. N = ids.numel().

    For S>1 the captured tensor is (B*S, T, D) per layer; we reshape to
    (S, B*T, D) and concat across batches.
    """
    L, S = cfg.n_layer, cfg.hyper_conn_n
    D = cfg.n_embd
    captured: dict[int, list[torch.Tensor]] = {i: [] for i in range(L)}

    def make_hook(li):
        def hook(_m, _inp, out):
            captured[li].append(out.detach())
        return hook

    handles = []
    for i, block in enumerate(model.transformer.h):
        handles.append(block.hc_mlp.register_forward_hook(make_hook(i)))

    try:
        for batch in ids:                  # (seq_len,)
            x = batch.unsqueeze(0).to(device)  # (1, T)
            _ = model(x)
    finally:
        for h in handles:
            h.remove()

    # Reshape to (L, S, N, D)
    N = ids.numel()
    out = torch.empty((L, S, N, D), dtype=torch.float16)
    for li in range(L):
        # captured[li] is a list of (S, T, D) tensors (per sequence)
        # — model returns (B*S, T, D), with B=1 → (S, T, D)
        chunks = [t.reshape(S, -1, D).cpu() for t in captured[li]]
        cat = torch.cat(chunks, dim=1).to(torch.float16)   # (S, N, D)
        out[li] = cat
    return out


# ---------- Linear AE (closed-form ridge regression, per pair) ----------
def fit_linear_ae(X_tr: torch.Tensor, Y_tr: torch.Tensor, ridge: float) -> torch.Tensor:
    """X, Y: (N, D). Returns W (D, D) with bias absorbed via prepending 1."""
    Xb = torch.cat([X_tr, torch.ones(X_tr.shape[0], 1, device=X_tr.device)], dim=1)  # (N, D+1)
    A = Xb.T @ Xb                                    # (D+1, D+1)
    A.diagonal().add_(ridge * X_tr.shape[0])
    B = Xb.T @ Y_tr                                  # (D+1, D)
    W = torch.linalg.solve(A, B)                     # (D+1, D)
    return W


def apply_linear(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    Xb = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
    return Xb @ W


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Coefficient of determination across all dims (variance-explained)."""
    ss_res = ((target - pred) ** 2).sum().item()
    ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum().item()
    return 1.0 - ss_res / max(ss_tot, 1e-12)


# ---------- MLP AE (1 hidden layer) ----------
class MLPAE(nn.Module):
    def __init__(self, D: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, hidden), nn.GELU(),
            nn.Linear(hidden, D),
        )

    def forward(self, x):
        return self.net(x)


def fit_mlp_ae(X_tr, Y_tr, X_te, Y_te, D: int, *, hidden: int, epochs: int,
               batch_size: int, lr: float) -> float:
    model = MLPAE(D, hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = X_tr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xb, yb = X_tr[idx], Y_tr[idx]
            opt.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(X_te)
    return r2_score(pred, Y_te)


# ---------- Hyper-conn parameter norms ----------
def per_layer_alpha_beta_norms(model, cfg, ids: torch.Tensor) -> list[dict]:
    """Returns one dict per layer: static + dynamic param norms for the MLP block's
    hyper-connection wrapper (hc_mlp), since SAEs and the AE target stream both
    relate to the MLP delta path."""
    rows = []
    L, S = cfg.n_layer, cfg.hyper_conn_n

    # Capture dynamic_alpha/dynamic_beta values per layer by hooking the
    # hc_mlp.dynamic_alpha_fn / dynamic_beta_fn parameters' use. Easier: just
    # store the static params + parameter magnitudes; the dynamic functions are
    # nn.Parameters used as projection matrices, so report their Frobenius norm.
    for li, block in enumerate(model.transformer.h):
        hc = block.hc_mlp  # ManifoldConstrainedHyperConnections wrapper
        row = {"layer": li}
        if hasattr(hc, "static_alpha"):
            row["static_alpha_fro"] = float(hc.static_alpha.detach().norm().item())
            row["static_alpha_mean"] = float(hc.static_alpha.detach().abs().mean().item())
        if hasattr(hc, "static_beta"):
            row["static_beta_fro"] = float(hc.static_beta.detach().norm().item())
            row["static_beta_mean"] = float(hc.static_beta.detach().abs().mean().item())
        if hasattr(hc, "dynamic_alpha_fn"):
            row["dynamic_alpha_fro"] = float(hc.dynamic_alpha_fn.detach().norm().item())
        if hasattr(hc, "dynamic_beta_fn"):
            row["dynamic_beta_fro"] = float(hc.dynamic_beta_fn.detach().norm().item())
        if hasattr(hc, "h_post_scale"):
            row["h_post_scale"] = float(hc.h_post_scale.detach().item())
        if hasattr(hc, "pre_branch_scale"):
            row["pre_branch_scale"] = float(hc.pre_branch_scale.detach().item())
        if hasattr(hc, "residual_scale"):
            row["residual_scale"] = float(hc.residual_scale.detach().item())
        rows.append(row)
    return rows


# ---------- per-model driver ----------
def run_one_model(name: str, repo_id: str, ids: torch.Tensor):
    print(f"\n{'='*80}\n>>> {name}  ({repo_id})\n{'='*80}")
    model, cfg = load_model_from_repo(repo_id, device)
    L, S, D = cfg.n_layer, cfg.hyper_conn_n, cfg.n_embd
    print(f"L={L}  S={S}  D={D}  type={cfg.hyper_conn_type}")

    out_dir = RESULTS_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- alpha/beta extraction (one-shot, no forward needed) --
    ab = per_layer_alpha_beta_norms(model, cfg, ids)
    ab_csv = out_dir / "alpha_beta.csv"
    with open(ab_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ab[0].keys()))
        w.writeheader()
        w.writerows(ab)
    print(f"saved {ab_csv}")

    # -- capture stream activations --
    print("capturing post_mlp streams...")
    A = capture_post_mlp_streams(model, cfg, ids)   # (L, S, N, D) fp16
    N = A.shape[2]
    print(f"  shape={tuple(A.shape)}  (N={N} tokens)")

    # Free the model — AE training is on activations only.
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # -- per-layer fit linear + MLP AE for every (i, j) pair --
    r2_lin = np.full((L, S, S), np.nan, dtype=np.float32)
    r2_mlp = np.full((L, S, S), np.nan, dtype=np.float32)

    # SHUFFLED 80/20 split — Pile streams sequences in source order, so a
    # contiguous tail-split puts the test set in a different domain than the
    # train set. That artifact pushes R² strongly negative on layers where
    # streams encode source-specific features.
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(0))
    n_train = int(N * 0.8)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    for li in range(L):
        layer_acts = A[li].float().to(device)  # (S, N, D)
        for i in range(S):
            X = layer_acts[i]                  # (N, D)
            X_tr, X_te = X[train_idx], X[test_idx]
            for j in range(S):
                Y = layer_acts[j]
                Y_tr, Y_te = Y[train_idx], Y[test_idx]
                # Linear AE
                W = fit_linear_ae(X_tr, Y_tr, ridge=RIDGE)
                pred = apply_linear(W, X_te)
                r2_lin[li, i, j] = r2_score(pred, Y_te)
                # MLP AE
                r2_mlp[li, i, j] = fit_mlp_ae(
                    X_tr, Y_tr, X_te, Y_te, D=D,
                    hidden=D // 2, epochs=MLP_EPOCHS,
                    batch_size=MLP_BATCH, lr=MLP_LR,
                )
        del layer_acts
        if device == "cuda":
            torch.cuda.empty_cache()
        if li % 5 == 0:
            print(f"  layer {li:2d}: mean off-diag R²(lin)={float(np.nanmean([r2_lin[li, i, j] for i in range(S) for j in range(S) if i != j])):.3f}, "
                  f"R²(mlp)={float(np.nanmean([r2_mlp[li, i, j] for i in range(S) for j in range(S) if i != j])):.3f}")

    np.save(out_dir / "r2_linear.npy", r2_lin)
    np.save(out_dir / "r2_mlp.npy", r2_mlp)
    print(f"saved r2_linear.npy + r2_mlp.npy  shape={(L, S, S)}")

    # Cross-stream summary CSV — mean off-diagonal R² per layer.
    summary_path = out_dir / "cross_stream_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "mean_r2_linear_off_diag", "mean_r2_mlp_off_diag",
                    "min_r2_linear_off_diag", "min_r2_mlp_off_diag"])
        for li in range(L):
            off = [(i, j) for i in range(S) for j in range(S) if i != j]
            lin_vals = [r2_lin[li, i, j] for i, j in off]
            mlp_vals = [r2_mlp[li, i, j] for i, j in off]
            w.writerow([li,
                        f"{float(np.mean(lin_vals)):.6f}",
                        f"{float(np.mean(mlp_vals)):.6f}",
                        f"{float(np.min(lin_vals)):.6f}",
                        f"{float(np.min(mlp_vals)):.6f}"])
    print(f"saved {summary_path}")


def make_combined_figure():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for col, model in enumerate(["mhc", "mhc_lite"]):
        d = RESULTS_ROOT / model
        if not (d / "r2_linear.npy").exists():
            continue
        r2_lin = np.load(d / "r2_linear.npy")
        r2_mlp = np.load(d / "r2_mlp.npy")
        ab = list(csv.DictReader(open(d / "alpha_beta.csv")))
        L, S, _ = r2_lin.shape

        # Mean off-diag R² per layer
        off = [(i, j) for i in range(S) for j in range(S) if i != j]
        lin_mean = np.array([np.mean([r2_lin[li, i, j] for i, j in off]) for li in range(L)])
        mlp_mean = np.array([np.mean([r2_mlp[li, i, j] for i, j in off]) for li in range(L)])
        beta = np.array([float(r["static_beta_mean"]) for r in ab])
        dyn_alpha = np.array([float(r["dynamic_alpha_fro"]) for r in ab])

        ax_top = axes[0, col]
        ax_top.plot(range(L), lin_mean, label="R² linear AE", color="#2980b9", marker="o", markersize=3)
        ax_top.plot(range(L), mlp_mean, label="R² MLP AE",   color="#c0392b", marker="s", markersize=3)
        ax_top.set_title(f"{model}: cross-stream AE R² per layer (mean off-diag)", fontweight="bold")
        ax_top.set_xlabel("layer"); ax_top.set_ylabel("R²")
        ax_top.set_ylim(-0.05, 1.05)
        ax_top.grid(alpha=0.3); ax_top.legend(loc="lower right")

        ax_bot = axes[1, col]
        ax_bot2 = ax_bot.twinx()
        ax_bot.plot(range(L), beta, color="#7f8c8d", marker="^", markersize=3,
                    label="static_beta (mean |·|)")
        ax_bot2.plot(range(L), dyn_alpha, color="#27ae60", marker="v", markersize=3,
                     label="dynamic_alpha_fn (Frobenius)")
        ax_bot.set_title(f"{model}: hyper-conn (hc_mlp) param magnitudes", fontweight="bold")
        ax_bot.set_xlabel("layer"); ax_bot.set_ylabel("static_beta")
        ax_bot2.set_ylabel("dynamic_alpha_fn")
        ax_bot.grid(alpha=0.3)
        lns = ax_bot.lines + ax_bot2.lines
        ax_bot.legend(lns, [l.get_label() for l in lns], loc="upper right")

    fig.suptitle("Stream-AE accuracy vs hyper-connection parameter magnitudes (mhc & mhc_lite)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig_path = RESULTS_ROOT / "r2_vs_beta.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_path}")


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"loading corpus: {N_TOKENS} tokens (Dolma) ...")
    ids = get_token_corpus(N_TOKENS, seq_len=SEQ_LEN, source="dolma")
    print(f"  corpus shape: {ids.shape}")
    for spec in MODELS:
        run_one_model(spec["name"], spec["repo_id"], ids)
    make_combined_figure()


if __name__ == "__main__":
    main()

# %%
