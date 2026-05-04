# %%
"""Path-patching v1: decompose each head's effect on prev-token language-modeling
into DIRECT (writes to logits via residual skip) and INDIRECT (feeds into
downstream heads/MLPs) components.

For each (layer, head):
  1. Capture every block's output on the baseline forward.
  2. Ablate the head; freeze all subsequent block outputs to their baseline ⇒
     measure Δ NLL = the head's DIRECT effect on the final output.
  3. Compare to the head's TOTAL Δ NLL (plain ablation, already computed in
     head_finder.py / scores.csv).
  4. INDIRECT = TOTAL − DIRECT.

A "real prev-token head" should have:
  * pattern A[i, i-1] high on natural text (looks like a PT head)
  * INDIRECT effect high on natural text NLL (its main effect is via downstream
    consumers — typically induction / S-inhibition)
  * DIRECT effect ≈ 0 (PT heads don't write directly to logits)

A "vestigial PT-shaped" head has high pattern but low total / indirect.
A "load-bearing PT" head has high indirect Δ NLL.

Outputs:
  results/heads/prev_token/{model}/path_patch.csv
  results/analysis/prev_token_path_patch.png

Run:
    uv run python src/mhc_interp/path_patch_pt.py
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mhc_interp._loader import load_model_from_repo

device = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HEADS_ROOT = REPO_ROOT / "results" / "heads"
OUT_DIR = REPO_ROOT / "results" / "analysis"

MODELS = [
    {"name": "mhc",      "repo_id": "Realmbird/mhc-781m-mhc"},
    {"name": "mhc_lite", "repo_id": "Realmbird/mhc-781m-mhc-lite"},
]
MODEL_COLORS = {"mhc": "#c0392b", "mhc_lite": "#2980b9"}
TOP_K = 5


@contextmanager
def ablate_head_ctx(model, layer_i: int, head_i: int, hs: int):
    c_proj = model.transformer.h[layer_i].branch_attn[1].c_proj
    def pre_hook(_m, inputs):
        (x,) = inputs
        x = x.clone()
        x[..., head_i * hs:(head_i + 1) * hs] = 0.0
        return (x,)
    handle = c_proj.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def freeze_blocks_ctx(model, baseline_block_outputs: dict[int, torch.Tensor], from_layer: int):
    """Replace the forward output of every block at index ≥ from_layer with the
    cached baseline output. Used to isolate the DIRECT effect of an ablation at
    earlier layers — the modified residual cannot interact with downstream
    block computations."""
    handles = []
    L = len(model.transformer.h)
    for lj in range(from_layer, L):
        def make_hook(lj=lj):
            def hook(_m, _inp, _out):
                return baseline_block_outputs[lj]
            return hook
        handles.append(model.transformer.h[lj].register_forward_hook(make_hook()))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def nll_full_positions(model, ids: torch.Tensor) -> float:
    """Mean cross-entropy of next-token prediction across all positions."""
    logits, _ = model(ids, torch.zeros_like(ids))
    log_p = F.log_softmax(logits, dim=-1)
    targets = ids[:, 1:]
    log_p_pred = log_p[:, :-1, :]
    nll = -log_p_pred.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(nll.mean().item())


def capture_baseline_block_outputs(model, ids: torch.Tensor) -> tuple[float, dict[int, torch.Tensor]]:
    """Returns (baseline_nll, {layer_i: block output tensor})."""
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for li, block in enumerate(model.transformer.h):
        def make_hook(li=li):
            def hook(_m, _inp, out):
                captures[li] = out.detach().clone()
            return hook
        handles.append(block.register_forward_hook(make_hook()))
    try:
        baseline_nll = nll_full_positions(model, ids)
    finally:
        for h in handles:
            h.remove()
    return baseline_nll, captures


@torch.no_grad()
def direct_effect_grid(model, cfg, ids: torch.Tensor) -> tuple[float, np.ndarray]:
    """For every (l, h): ablate (l, h), freeze block outputs at l+1..L-1 to
    baseline, compute Δ NLL.  Returns (baseline_nll, direct_delta[L, H])."""
    L, H = cfg.n_layer, cfg.n_head
    hs = cfg.n_embd // H
    baseline_nll, baseline_blocks = capture_baseline_block_outputs(model, ids)
    direct = np.zeros((L, H), dtype=np.float32)
    for li in range(L):
        for hi in range(H):
            with ablate_head_ctx(model, li, hi, hs), \
                 freeze_blocks_ctx(model, baseline_blocks, from_layer=li + 1):
                nll = nll_full_positions(model, ids)
            direct[li, hi] = nll - baseline_nll
    return baseline_nll, direct


def render_thumbnails(slug: str, model: str, top_heads: list[tuple[int, int, str]],
                       attention_npy: Path, ax_row, total_arr, direct_arr,
                       indirect_arr, pattern_arr):
    """top_heads: list of (layer, head, label_for_subplot). attention_npy is
    the (L, H, T, T) tensor saved earlier for this (slug, model)."""
    A = np.load(attention_npy).astype(np.float32)
    tokens_meta_path = attention_npy.parent / "tokens.json"
    tokens = json.loads(tokens_meta_path.read_text())["tokens"]
    T = A.shape[-1]
    rows_canon = np.arange(1, T)
    cols_canon = np.arange(0, T - 1)
    for ax, (li, hi, _) in zip(ax_row, top_heads):
        P = A[li, hi]
        ax.imshow(P, cmap="viridis", vmin=0, vmax=1, aspect="equal")
        ax.plot(cols_canon, rows_canon, color="white", lw=0.5,
                alpha=0.5, linestyle="--")
        ax.set_xticks([])
        ax.set_yticks([])
        title = (
            f"L{li}h{hi}\n"
            f"pat={pattern_arr[li, hi]:.2f}  ind={indirect_arr[li, hi]:+.3f}\n"
            f"dir={direct_arr[li, hi]:+.3f}  tot={total_arr[li, hi]:+.3f}"
        )
        ax.set_title(title, fontsize=8.2, pad=3, color=MODEL_COLORS[model],
                     fontweight="bold")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    figs_payload = {}
    for spec in MODELS:
        name, repo_id = spec["name"], spec["repo_id"]
        print(f"\n>>> {name}")
        model, cfg = load_model_from_repo(repo_id, device)

        # Use the PT probe text from the existing scores file's prompt.json.
        prompt_text = json.loads((HEADS_ROOT / "prev_token" / "prompt.json").read_text())["probe"]["text"]
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        ids = torch.tensor([tok.encode(prompt_text)], device=device)

        baseline_nll, direct = direct_effect_grid(model, cfg, ids)
        print(f"  baseline NLL: {baseline_nll:.3f}")

        # Pull totals from the scores.csv we already wrote.
        df = pd.read_csv(HEADS_ROOT / "prev_token" / name / "scores.csv")
        L, H = cfg.n_layer, cfg.n_head
        total = np.zeros((L, H), dtype=np.float32)
        pattern = np.zeros((L, H), dtype=np.float32)
        for r in df.itertuples(index=False):
            total[int(r.layer), int(r.head)] = float(r.ablation_delta_nll)
            pattern[int(r.layer), int(r.head)] = float(r.pattern_score)
        indirect = total - direct

        # Save extended CSV.
        out_rows = []
        for li in range(L):
            for hi in range(H):
                out_rows.append({
                    "layer": li, "head": hi,
                    "pattern_score": float(pattern[li, hi]),
                    "total_delta_nll": float(total[li, hi]),
                    "direct_delta_nll": float(direct[li, hi]),
                    "indirect_delta_nll": float(indirect[li, hi]),
                })
        df_out = pd.DataFrame(out_rows)
        out_csv = HEADS_ROOT / "prev_token" / name / "path_patch.csv"
        df_out.to_csv(out_csv, index=False)
        print(f"  saved {out_csv}")

        # Stash arrays for the figure step.
        figs_payload[name] = {
            "pattern": pattern, "total": total,
            "direct": direct, "indirect": indirect,
            "attention_npy": HEADS_ROOT / "prev_token" / name / "attention.npy",
            "df": df_out,
        }

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ------------- Figure -------------
    fig, axes = plt.subplots(
        len(MODELS) * 2, TOP_K,
        figsize=(13, 9.5),
        gridspec_kw={"hspace": 0.85, "wspace": 0.18},
    )

    for mi, name in enumerate(["mhc", "mhc_lite"]):
        p = figs_payload[name]
        df = p["df"]

        # Top-K by INDIRECT effect (the "load-bearing PT" candidates).
        top_indirect = df.sort_values("indirect_delta_nll", ascending=False).head(TOP_K)
        # Top-K by PATTERN (the "looks like PT" candidates).
        top_pattern = df.sort_values("pattern_score", ascending=False).head(TOP_K)

        ind_heads = [(int(r["layer"]), int(r["head"]), "ind") for _, r in top_indirect.iterrows()]
        pat_heads = [(int(r["layer"]), int(r["head"]), "pat") for _, r in top_pattern.iterrows()]

        render_thumbnails(
            "prev_token", name, ind_heads, p["attention_npy"],
            axes[2 * mi], p["total"], p["direct"], p["indirect"], p["pattern"],
        )
        axes[2 * mi, 0].set_ylabel(
            f"{name}\ntop-5 by INDIRECT",
            fontsize=10, fontweight="bold",
            color=MODEL_COLORS[name], labelpad=12,
        )

        render_thumbnails(
            "prev_token", name, pat_heads, p["attention_npy"],
            axes[2 * mi + 1], p["total"], p["direct"], p["indirect"], p["pattern"],
        )
        axes[2 * mi + 1, 0].set_ylabel(
            f"{name}\ntop-5 by PATTERN",
            fontsize=10, fontweight="bold",
            color=MODEL_COLORS[name], labelpad=12,
        )

    fig.suptitle(
        "Prev-token heads — load-bearing (indirect Δ NLL) vs pattern-shaped",
        fontsize=14, fontweight="bold", y=1.0,
    )
    fig.text(
        0.5, -0.01,
        "Each thumbnail = attention pattern of one head (white dashed line = canonical PT stripe). "
        "Annotation: pat = stripe-mass; ind = indirect Δ NLL via downstream consumers; "
        "dir = direct Δ NLL through skip-to-unembed only; tot = full ablation Δ NLL. "
        "Load-bearing PT heads should have ind ≫ dir.",
        ha="center", fontsize=8.5, style="italic", color="#444",
    )
    fig.tight_layout()
    out_path = OUT_DIR / "prev_token_path_patch.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    # Print summary stats for write-up.
    print("\nSummary — top-5 by INDIRECT vs by PATTERN, with overlap:")
    for name in ["mhc", "mhc_lite"]:
        df = figs_payload[name]["df"]
        ind_df = df.sort_values("indirect_delta_nll", ascending=False).iloc[:TOP_K]
        pat_df = df.sort_values("pattern_score",       ascending=False).iloc[:TOP_K]
        top_ind = set(zip(ind_df["layer"].astype(int), ind_df["head"].astype(int)))
        top_pat = set(zip(pat_df["layer"].astype(int), pat_df["head"].astype(int)))
        overlap = top_ind & top_pat
        print(f"  {name:10s}  top-5 indirect ∩ top-5 pattern = {len(overlap)}  ({sorted(overlap)})")
        print(f"             top-5 by indirect:  {sorted(top_ind)}")
        print(f"             top-5 by pattern :  {sorted(top_pat)}")


if __name__ == "__main__":
    main()

# %%
