# %%
"""Option A — head-to-head path-patching at attention-pattern resolution.

For each load-bearing PT candidate `h_S` (top-3 by indirect ΔNLL from
path_patch_pt.py), ablate it and capture every downstream head's attention
pattern. The heads whose patterns shift most (Frobenius norm of Δ) are the
*consumers* of h_S — the downstream heads that read what h_S writes.

Output:
  results/analysis/path_patch_consumers_{mhc,mhc_lite}.png
  results/heads/prev_token/{model}/consumer_attribution.npz

Run:
    uv run python src/mhc_interp/path_patch_consumers.py
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoTokenizer

from mhc_interp._loader import attn_from_qkv, load_model_from_repo

device = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HEADS_ROOT = REPO_ROOT / "results" / "heads"
OUT_DIR = REPO_ROOT / "results" / "analysis"

MODELS = [
    {"name": "mhc",      "repo_id": "Realmbird/mhc-781m-mhc"},
    {"name": "mhc_lite", "repo_id": "Realmbird/mhc-781m-mhc-lite"},
]
MODEL_COLORS = {"mhc": "#c0392b", "mhc_lite": "#2980b9"}
N_CANDIDATES = 3
N_CONSUMERS = 3


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


@torch.no_grad()
def capture_all_attention(model, ids: torch.Tensor, n_head: int) -> np.ndarray:
    """Run forward, return (L, H, T, T) attention as fp32 numpy."""
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for li, block in enumerate(model.transformer.h):
        attn_module = block.branch_attn[1]
        def make_hook(li=li):
            def hook(_m, _inp, out):
                captures[li] = out.detach()
            return hook
        handles.append(attn_module.c_attn.register_forward_hook(make_hook()))
    try:
        _ = model(ids)
    finally:
        for h in handles:
            h.remove()
    L = len(model.transformer.h)
    out = []
    for li in range(L):
        a = attn_from_qkv(captures[li], n_head)  # (1, H, T, T)
        out.append(a[0].cpu().numpy())
    return np.stack(out).astype(np.float32)  # (L, H, T, T)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("gpt2")
    prompt_text = json.loads((HEADS_ROOT / "prev_token" / "prompt.json").read_text())["probe"]["text"]

    for spec in MODELS:
        name, repo_id = spec["name"], spec["repo_id"]
        print(f"\n>>> {name}")
        model, cfg = load_model_from_repo(repo_id, device)
        L, H = cfg.n_layer, cfg.n_head
        hs = cfg.n_embd // H
        ids = torch.tensor([tok.encode(prompt_text)], device=device)
        T = ids.shape[1]

        # Top-N_CANDIDATES PT candidates by indirect ΔNLL.
        df = pd.read_csv(HEADS_ROOT / "prev_token" / name / "path_patch.csv")
        cand_df = df.sort_values("indirect_delta_nll", ascending=False).iloc[:N_CANDIDATES]
        candidates = [(int(r["layer"]), int(r["head"])) for _, r in cand_df.iterrows()]
        print(f"  candidates (top-{N_CANDIDATES} by indirect ΔNLL): {candidates}")

        # Baseline attention.
        baseline_attn = capture_all_attention(model, ids, H)
        print(f"  baseline attention captured: {baseline_attn.shape}")

        results = {}
        for (l_S, h_S) in candidates:
            with ablate_head_ctx(model, l_S, h_S, hs):
                abl_attn = capture_all_attention(model, ids, H)
            delta = baseline_attn - abl_attn  # (L, H, T, T)
            delta_norm = np.linalg.norm(delta.reshape(L, H, -1), axis=-1)  # (L, H)
            # Causal: ablating l_S has no effect at layers ≤ l_S in this single-pass
            # forward (each block's attention only depends on inputs at or below).
            delta_norm[:l_S + 1] = 0.0
            results[(l_S, h_S)] = {
                "delta_norm": delta_norm,
                "abl_attn": abl_attn,
            }
            print(f"  L{l_S}h{h_S}: max-Δ-norm={delta_norm.max():.3f} at "
                  f"L{int(np.argmax(delta_norm) // H)}h{int(np.argmax(delta_norm) % H)}")

        # Save the raw deltas per candidate.
        npz_payload = {"baseline_attention": baseline_attn, "tokens": np.array(prompt_text)}
        for (l_S, h_S), v in results.items():
            tag = f"L{l_S}H{h_S}"
            npz_payload[f"{tag}_delta_norm"] = v["delta_norm"]
            npz_payload[f"{tag}_ablated_attention"] = v["abl_attn"]
        np.savez_compressed(
            HEADS_ROOT / "prev_token" / name / "consumer_attribution.npz",
            **npz_payload,
        )
        print(f"  saved consumer_attribution.npz")

        # ---- Figure: per-candidate row, columns = source / Δ heatmap / top-3 consumers ----
        fig, axes = plt.subplots(
            N_CANDIDATES, 2 + N_CONSUMERS,
            figsize=(15, 3.4 * N_CANDIDATES),
            gridspec_kw={"hspace": 0.55, "wspace": 0.32,
                          "width_ratios": [1, 1.2, 1, 1, 1]},
        )
        if N_CANDIDATES == 1:
            axes = axes[None, :]
        cmap = LinearSegmentedColormap.from_list("delta", ["#f7f7f7", "#fdcc8a", "#e34a33", "#7f0000"])

        for ri, ((l_S, h_S), v) in enumerate(results.items()):
            # col 0: source head's pattern
            ax = axes[ri, 0]
            P = baseline_attn[l_S, h_S]
            ax.imshow(P, cmap="viridis", vmin=0, vmax=1, aspect="equal")
            # canonical PT stripe
            xs = np.arange(0, T - 1); ys = np.arange(1, T)
            ax.plot(xs, ys, color="white", lw=0.5, alpha=0.5, linestyle="--")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"source\nL{l_S}h{h_S}",
                         fontsize=11, fontweight="bold", pad=4,
                         color=MODEL_COLORS[name])
            if ri == 0:
                ax.set_ylabel("ablated\nthis →",
                              fontsize=9, color="#666",
                              labelpad=10)

            # col 1: Δ heatmap over (l_R, h_R)
            ax = axes[ri, 1]
            DN = v["delta_norm"]  # (L, H)
            im = ax.imshow(DN, cmap=cmap, aspect="auto")
            ax.set_xlabel("head", fontsize=9)
            ax.set_ylabel("layer", fontsize=9)
            ax.set_title("downstream attention shift\n(ΔF norm per consumer)",
                         fontsize=10, fontweight="bold", pad=4)
            ax.set_xticks([0, 5, 10, 15, 19])
            ax.set_yticks([0, 5, 10, 15, 20, 25, 30, 35])
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

            # cols 2..2+N_CONSUMERS: top consumer patterns (baseline)
            flat_idx = np.argsort(-DN.flatten())
            top_consumers = [(int(idx // H), int(idx % H)) for idx in flat_idx[:N_CONSUMERS]]
            for ci, (l_R, h_R) in enumerate(top_consumers):
                ax = axes[ri, 2 + ci]
                P_R = baseline_attn[l_R, h_R]
                ax.imshow(P_R, cmap="viridis", vmin=0, vmax=1, aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                dn_val = float(DN[l_R, h_R])
                ax.set_title(f"top-{ci + 1} consumer\nL{l_R}h{h_R}  (Δ={dn_val:.2f})",
                             fontsize=9.5, pad=4)

        fig.suptitle(
            f"{name} — load-bearing PT candidates and their downstream attention consumers",
            fontsize=13, fontweight="bold", y=1.0,
            color=MODEL_COLORS[name],
        )
        fig.text(
            0.5, -0.005,
            "Col 1: source PT candidate. Col 2: Frobenius norm of attention change at every (layer, head) "
            "downstream — reds = strongly affected. "
            "Cols 3-5: baseline attention of the 3 most-affected consumer heads.",
            ha="center", fontsize=8.5, style="italic", color="#444",
        )
        out_path = OUT_DIR / f"path_patch_consumers_{name}.png"
        fig.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

# %%
