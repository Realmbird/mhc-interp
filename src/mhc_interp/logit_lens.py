# %%
"""Logit lens for the MHC-780M model from wgpeng/mhc-780m.

For each transformer block we capture:
  - attn branch output        (B, T, D) — single mixed view
  - mlp branch output         (B, T, D) — single mixed view
  - post-attn residual streams (B, S, T, D)
  - post-mlp residual streams  (B, S, T, D)

Logit lens = lm_head(ln_f(x)). We apply it to:
  * each branch output (attn / mlp) per layer
  * each individual stream (s_i) per layer
  * every subset of streams (sum then lens) per layer — 2^S - 1 = 15 subsets

Visualization style matches cywinski/codi/experiments/3_logit_lens_latents.py:
heatmap of top-1 prob with the top-1 token printed inside each cell.
"""
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

sys.path.insert(0, "/home/chriskino/mhc-lite")
from model import GPT, GPTConfig

# %% Load
ckpt_path = hf_hub_download(repo_id="wgpeng/mhc-780m", filename="best_ckpt_large.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = GPTConfig(**ckpt["model_args"])
model = GPT(cfg)
sd = ckpt["model"]
for k in list(sd.keys()):
    if k.startswith("_orig_mod."):
        sd[k[len("_orig_mod."):]] = sd.pop(k)
model.load_state_dict(sd)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

S = cfg.hyper_conn_n  # 4
L = cfg.n_layer       # 36
tok = AutoTokenizer.from_pretrained("gpt2")

# %% Hooks: capture every layer's branch outs and post-block residual streams
captures: dict[tuple[int, str], torch.Tensor] = {}

def _save(key):
    def hook(_m, _inp, out):
        captures[key] = out.detach()
    return hook

handles = []
for i, block in enumerate(model.transformer.h):
    handles.append(block.hc_attn.branch.register_forward_hook(_save((i, "attn_out"))))
    handles.append(block.hc_mlp.branch.register_forward_hook(_save((i, "mlp_out"))))
    handles.append(block.hc_attn.register_forward_hook(_save((i, "post_attn"))))
    handles.append(block.hc_mlp.register_forward_hook(_save((i, "post_mlp"))))

# %% Run a forward pass
PROMPT = "The capital of France is"
ids = torch.tensor([tok.encode(PROMPT)], dtype=torch.long, device=device)
with torch.no_grad():
    _ = model(ids)
for h in handles:
    h.remove()

# %% Lens helpers
@torch.no_grad()
def lens(x: torch.Tensor) -> torch.Tensor:
    """x: (..., D). Returns logits over vocab."""
    return model.lm_head(model.transformer.ln_f(x))

@torch.no_grad()
def topk(x: torch.Tensor, k: int = 1, position: int = -1):
    """Return top-k token ids and probs at given seq position. x: (B, T, D) or (T, D)."""
    logits = lens(x)
    if logits.dim() == 3:
        logits = logits[0, position]
    else:
        logits = logits[position]
    probs = F.softmax(logits, dim=-1)
    p, ix = probs.topk(k, dim=-1)
    return ix.cpu().tolist(), p.cpu().tolist()

def split_streams(x: torch.Tensor) -> torch.Tensor:
    """(B*S, T, D) -> (B, S, T, D)."""
    Bs, T, D = x.shape
    return x.reshape(Bs // S, S, T, D)

# %% Build matrices for visualization
def build_matrix(column_specs):
    """column_specs: list of (label, callable(layer_idx) -> tensor (B, T, D)).
    Returns (probs[L, C], tokens[L][C], labels[C])."""
    C = len(column_specs)
    probs = np.zeros((L, C), dtype=np.float32)
    tokens = [[""] * C for _ in range(L)]
    for layer_i in range(L):
        for c, (_, fn) in enumerate(column_specs):
            x = fn(layer_i)
            ids_, ps_ = topk(x, k=1)
            probs[layer_i, c] = ps_[0]
            tokens[layer_i][c] = tok.decode([ids_[0]])
    labels = [lbl for lbl, _ in column_specs]
    return probs, tokens, labels


def heatmap(probs, tokens, labels, title, out_path, figsize=None):
    """Codi-style heatmap: YlOrRd, top-1 token annotated in each cell."""
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
    return fig


# Output dir
results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "logit_lens"
results_dir.mkdir(parents=True, exist_ok=True)

# %% Figure 1 — branches + per-stream + full reduce
def col_attn(i):  return captures[(i, "attn_out")]
def col_mlp(i):   return captures[(i, "mlp_out")]
def col_reduce_mlp(i):  return split_streams(captures[(i, "post_mlp")]).sum(dim=1)
def col_reduce_attn(i): return split_streams(captures[(i, "post_attn")]).sum(dim=1)
def col_stream(s):
    return lambda i, s=s: split_streams(captures[(i, "post_mlp")])[:, s]

fig1_specs = [
    ("attn_out", col_attn),
    ("mlp_out", col_mlp),
    ("Σ post_attn", col_reduce_attn),
    ("Σ post_mlp", col_reduce_mlp),
] + [(f"s{s} (post_mlp)", col_stream(s)) for s in range(S)]

probs1, tokens1, labels1 = build_matrix(fig1_specs)
heatmap(
    probs1, tokens1, labels1,
    title=f"MHC-780M logit lens — {PROMPT!r} | branches + per-stream",
    out_path=results_dir / "fig1_branches_and_streams.png",
)

# %% Figure 2 — every subset of streams, post-MLP
subsets = [c for k in range(1, S + 1) for c in itertools.combinations(range(S), k)]
sub_labels = ["{" + ",".join(map(str, c)) + "}" for c in subsets]

def col_subset_postmlp(combo):
    return lambda i, combo=combo: split_streams(captures[(i, "post_mlp")])[:, list(combo)].sum(dim=1)

fig2_specs = [(lbl, col_subset_postmlp(c)) for lbl, c in zip(sub_labels, subsets)]
probs2, tokens2, labels2 = build_matrix(fig2_specs)
heatmap(
    probs2, tokens2, labels2,
    title=f"MHC-780M logit lens — {PROMPT!r} | post-MLP, all stream subsets",
    out_path=results_dir / "fig2_subsets_post_mlp.png",
)

# %% Figure 3 — every subset of streams, post-ATTN
def col_subset_postattn(combo):
    return lambda i, combo=combo: split_streams(captures[(i, "post_attn")])[:, list(combo)].sum(dim=1)

fig3_specs = [(lbl, col_subset_postattn(c)) for lbl, c in zip(sub_labels, subsets)]
probs3, tokens3, labels3 = build_matrix(fig3_specs)
heatmap(
    probs3, tokens3, labels3,
    title=f"MHC-780M logit lens — {PROMPT!r} | post-ATTN, all stream subsets",
    out_path=results_dir / "fig3_subsets_post_attn.png",
)

# %% Optional: top-K text table for the per-stream view (compact)
TOP_K = 5
print("=" * 80)
print(f"Prompt: {PROMPT!r}")
print(f"Top-{TOP_K} tokens at last position, per layer × view (post_mlp residual)")
print("=" * 80)
print(f"{'L':>3} | {'attn_out':<28} | {'mlp_out':<28} | " +
      " | ".join(f"{f's{s}':<28}" for s in range(S)))

for layer_i in range(L):
    cells = []
    for spec_fn in (col_attn, col_mlp, *(col_stream(s) for s in range(S))):
        ids_, ps_ = topk(spec_fn(layer_i), k=TOP_K)
        cell = " ".join(f"{repr(tok.decode([t]))[1:-1][:8]}({p:.2f})" for t, p in zip(ids_, ps_))
        cells.append(cell[:28])
    print(f"{layer_i:>3} | " + " | ".join(f"{c:<28}" for c in cells))

# %%
plt.show()

# %%
