# %%
"""Logit lens across the three Realmbird/mhc-781m-* variants.

Models (https://huggingface.co/collections/Realmbird/mhc-model-diff):
  - Realmbird/mhc-781m-residual   (hyper_conn_type="none",     n=1)
  - Realmbird/mhc-781m-mhc        (hyper_conn_type="mhc",      n=4)
  - Realmbird/mhc-781m-mhc-lite   (hyper_conn_type="mhc_lite", n=4)

For each transformer block we capture:
  - attn branch output
  - mlp branch output
  - post-attn residual (B,S,T,D) when S>1, else (B,T,D)
  - post-mlp  residual (B,S,T,D) when S>1, else (B,T,D)

Logit lens = lm_head(ln_f(x)). Applied to:
  * each branch output (attn / mlp) per layer
  * each individual stream (s_i) per layer
  * every subset of streams (sum then lens) per layer — 2^S - 1 subsets

Visualization: heatmap of top-1 prob with the top-1 token printed inside each cell.
"""
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from mhc_interp._loader import load_model_from_repo

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("gpt2")

# Common prompt that all three checkpoints agree on at the final layer.
# (residual / mhc / mhc-lite all top-1 ' floor' here — see greedy probe in README.)
# Picked over "The capital of France is" because at 781M / ~655M tokens these
# checkpoints don't reliably surface " Paris"; they produce structural filler
# (" located", " situated") instead, which makes the lens noisier to read.
PROMPT = "The cat sat on the"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = REPO_ROOT / "results" / "logit_lens"

MODELS = [
    {"name": "residual", "repo_id": "Realmbird/mhc-781m-residual"},
    {"name": "mhc",      "repo_id": "Realmbird/mhc-781m-mhc"},
    {"name": "mhc_lite", "repo_id": "Realmbird/mhc-781m-mhc-lite"},
]

# %% Lens helpers (closed over a specific loaded model)
def make_lens(model):
    @torch.no_grad()
    def lens(x: torch.Tensor) -> torch.Tensor:
        return model.lm_head(model.transformer.ln_f(x))

    @torch.no_grad()
    def topk(x: torch.Tensor, k: int = 1, position: int = -1):
        logits = lens(x)
        if logits.dim() == 3:
            logits = logits[0, position]
        else:
            logits = logits[position]
        probs = F.softmax(logits, dim=-1)
        p, ix = probs.topk(k, dim=-1)
        return ix.cpu().tolist(), p.cpu().tolist()

    return lens, topk


def split_streams(x: torch.Tensor, S: int) -> torch.Tensor:
    """(B*S, T, D) -> (B, S, T, D) when S>1; else add a dummy stream dim."""
    if S == 1:
        return x.unsqueeze(1)  # (B, 1, T, D)
    Bs, T, D = x.shape
    return x.reshape(Bs // S, S, T, D)


# %% Figure builder
def build_matrix(L, column_specs, topk, K: int = 5):
    """For each layer × view, store top-1 prob + token (for the heatmap) and
    top-K ids/probs (for the Streamlit drill-down)."""
    C = len(column_specs)
    probs = np.zeros((L, C), dtype=np.float32)
    tokens = [[""] * C for _ in range(L)]
    topk_ids = np.zeros((L, C, K), dtype=np.int32)
    topk_probs = np.zeros((L, C, K), dtype=np.float32)
    for layer_i in range(L):
        for c, (_, fn) in enumerate(column_specs):
            x = fn(layer_i)
            ids_, ps_ = topk(x, k=K)
            probs[layer_i, c] = ps_[0]
            tokens[layer_i][c] = tok.decode([ids_[0]])
            topk_ids[layer_i, c] = ids_
            topk_probs[layer_i, c] = ps_
    labels = [lbl for lbl, _ in column_specs]
    return probs, tokens, labels, topk_ids, topk_probs


def heatmap(probs, tokens, labels, title, out_path, figsize=None):
    rows, cols = probs.shape
    if figsize is None:
        figsize = (max(8, cols * 1.0), max(8, rows * 0.32))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(probs, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Top-1 probability", rotation=270, labelpad=15, fontsize=12)
    for i in range(rows):
        for j in range(cols):
            t = tokens[i][j] or ""
            disp = repr(t)[1:-1]
            if len(disp) > 10:
                disp = disp[:9] + "…"
            ax.text(
                j, i, disp,
                ha="center", va="center",
                color="white" if probs[i, j] > 0.5 else "black",
                fontsize=7,
            )
    ax.set_xlabel("View", fontsize=12, fontweight="bold")
    ax.set_ylabel("Layer", fontsize=12, fontweight="bold")
    ax.set_xticks(range(cols))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(rows))
    ax.set_yticklabels([str(i) for i in range(rows)], fontsize=8)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")
    plt.close(fig)


# %% Run a single model end-to-end
def run_model(name: str, repo_id: str):
    print(f"\n{'=' * 80}\n>>> {name}  ({repo_id})\n{'=' * 80}")
    model, cfg = load_model_from_repo(repo_id, device)
    S, L = cfg.hyper_conn_n, cfg.n_layer
    print(f"S={S}  L={L}  type={cfg.hyper_conn_type}")

    captures: dict[tuple[int, str], torch.Tensor] = {}

    def _save(key):
        def hook(_m, _inp, out):
            captures[key] = out.detach()
        return hook

    handles = []
    for i, block in enumerate(model.transformer.h):
        # Each repo's Block has hc_attn / hc_mlp wrappers (Residual for "none", MHC[Lite] otherwise).
        handles.append(block.hc_attn.branch.register_forward_hook(_save((i, "attn_out"))))
        handles.append(block.hc_mlp.branch.register_forward_hook(_save((i, "mlp_out"))))
        handles.append(block.hc_attn.register_forward_hook(_save((i, "post_attn"))))
        handles.append(block.hc_mlp.register_forward_hook(_save((i, "post_mlp"))))

    ids = torch.tensor([tok.encode(PROMPT)], dtype=torch.long, device=device)
    with torch.no_grad():
        _ = model(ids)
    for h in handles:
        h.remove()

    _, topk = make_lens(model)
    out_dir = RESULTS_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: branches + per-stream + sum-reduce
    def col_attn(i):  return captures[(i, "attn_out")]
    def col_mlp(i):   return captures[(i, "mlp_out")]
    def col_reduce_attn(i): return split_streams(captures[(i, "post_attn")], S).sum(dim=1)
    def col_reduce_mlp(i):  return split_streams(captures[(i, "post_mlp")],  S).sum(dim=1)
    def col_stream(s):
        return lambda i, s=s: split_streams(captures[(i, "post_mlp")], S)[:, s]

    fig1_specs = [
        ("attn_out", col_attn),
        ("mlp_out",  col_mlp),
        ("Σ post_attn", col_reduce_attn),
        ("Σ post_mlp",  col_reduce_mlp),
    ] + [(f"s{s} (post_mlp)", col_stream(s)) for s in range(S)]

    figure_data = {}  # collected for the npz dump below

    probs1, tokens1, labels1, ids1, p1 = build_matrix(L, fig1_specs, topk)
    heatmap(
        probs1, tokens1, labels1,
        title=f"{name} ({repo_id}) — {PROMPT!r} | branches + per-stream",
        out_path=out_dir / "fig1_branches_and_streams.png",
    )
    figure_data["fig1"] = dict(probs=probs1, tokens=tokens1, labels=labels1,
                               topk_ids=ids1, topk_probs=p1)

    # ---- Figures 2 & 3: subset-sums (only meaningful when S > 1)
    if S > 1:
        subsets = [c for k in range(1, S + 1) for c in itertools.combinations(range(S), k)]
        sub_labels = ["{" + ",".join(map(str, c)) + "}" for c in subsets]

        def col_subset(stage, combo):
            return lambda i, combo=combo, stage=stage: split_streams(
                captures[(i, stage)], S
            )[:, list(combo)].sum(dim=1)

        for stage, fname, suffix, key in [
            ("post_mlp",  "fig2_subsets_post_mlp.png",  "post-MLP",  "fig2"),
            ("post_attn", "fig3_subsets_post_attn.png", "post-ATTN", "fig3"),
        ]:
            specs = [(lbl, col_subset(stage, c)) for lbl, c in zip(sub_labels, subsets)]
            probs, tokens, labels, ids_k, p_k = build_matrix(L, specs, topk)
            heatmap(
                probs, tokens, labels,
                title=f"{name} ({repo_id}) — {PROMPT!r} | {suffix}, all stream subsets",
                out_path=out_dir / fname,
            )
            figure_data[key] = dict(probs=probs, tokens=tokens, labels=labels,
                                    topk_ids=ids_k, topk_probs=p_k)
    else:
        print(f"[{name}] S=1 — skipping subset figures (only one stream).")

    # ---- Raw data dumps consumed by the Streamlit viewer.
    npz_payload = {}
    meta = {
        "model_name": name, "repo_id": repo_id, "hyper_conn_type": cfg.hyper_conn_type,
        "S": S, "L": L, "prompt": PROMPT, "figures": {},
    }
    for fig_key, d in figure_data.items():
        npz_payload[f"{fig_key}_probs"] = d["probs"]
        npz_payload[f"{fig_key}_topk_ids"] = d["topk_ids"]
        npz_payload[f"{fig_key}_topk_probs"] = d["topk_probs"]
        meta["figures"][fig_key] = {
            "labels": d["labels"],
            "top1_tokens": d["tokens"],  # decoded strings, [L][C]
        }
    np.savez_compressed(out_dir / "lens_data.npz", **npz_payload)
    with open(out_dir / "lens_meta.json", "w") as f:
        # ids2tokens for top-K — pre-decode so the app doesn't need the tokenizer.
        topk_tokens = {}
        for fig_key, d in figure_data.items():
            ids = d["topk_ids"]  # (L, C, K)
            decoded = [[[tok.decode([int(t)]) for t in ids[li, ci]]
                        for ci in range(ids.shape[1])]
                       for li in range(ids.shape[0])]
            topk_tokens[fig_key] = decoded
        meta["topk_tokens"] = topk_tokens
        json.dump(meta, f, indent=2)
    print(f"saved {out_dir / 'lens_data.npz'}  +  {out_dir / 'lens_meta.json'}")

    # ---- Top-K text dump
    TOP_K = 5
    txt_path = out_dir / "topk.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Model: {name}  ({repo_id})\nPrompt: {PROMPT!r}\n")
        f.write(f"Top-{TOP_K} tokens at last position, per layer × view\n")
        f.write("=" * 80 + "\n")
        header = f"{'L':>3} | {'attn_out':<28} | {'mlp_out':<28} | " + \
                 " | ".join(f"{f's{s}':<28}" for s in range(S))
        f.write(header + "\n")
        for layer_i in range(L):
            cells = []
            for spec_fn in (col_attn, col_mlp, *(col_stream(s) for s in range(S))):
                ids_, ps_ = topk(spec_fn(layer_i), k=TOP_K)
                cell = " ".join(f"{repr(tok.decode([t]))[1:-1][:8]}({p:.2f})" for t, p in zip(ids_, ps_))
                cells.append(cell[:28])
            f.write(f"{layer_i:>3} | " + " | ".join(f"{c:<28}" for c in cells) + "\n")
    print(f"saved {txt_path}")

    # Free GPU before next model
    del model, captures
    if device == "cuda":
        torch.cuda.empty_cache()


# %% Drive
if __name__ == "__main__":
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    for spec in MODELS:
        run_model(spec["name"], spec["repo_id"])

# %%
