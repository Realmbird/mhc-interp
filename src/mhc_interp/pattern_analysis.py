# %%
"""Quantitative pattern analysis on top of logit_lens captures.

Computes per-layer × per-stream:
  1. Entropy of the lens distribution (how diffuse the prediction is)
  2. Probability mass on a specific target token (e.g. ' Paris')
  3. Pair-wise cosine similarity between streams in the residual space
  4. Layer where streams first agree vs disagree (top-1 set size across streams)

Saves heatmaps and prints a summary.
"""
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

S, L = cfg.hyper_conn_n, cfg.n_layer
tok = AutoTokenizer.from_pretrained("gpt2")

# %% Hooks
captures = {}
def _save(key):
    def hook(_m, _inp, out): captures[key] = out.detach()
    return hook

handles = []
for i, block in enumerate(model.transformer.h):
    handles.append(block.hc_attn.register_forward_hook(_save((i, "post_attn"))))
    handles.append(block.hc_mlp.register_forward_hook(_save((i, "post_mlp"))))

PROMPT = "The capital of France is"
TARGET = " Paris"
ids = torch.tensor([tok.encode(PROMPT)], dtype=torch.long, device=device)
with torch.no_grad():
    _ = model(ids)
for h in handles:
    h.remove()

target_id = tok.encode(TARGET)[0]
print(f"Prompt: {PROMPT!r}, target token {TARGET!r} -> id {target_id}")

# %% Helpers
@torch.no_grad()
def lens_dist(x):
    """x: (B, T, D). Returns probs at LAST position. Shape (V,)."""
    logits = model.lm_head(model.transformer.ln_f(x))[0, -1]
    return F.softmax(logits, dim=-1)

def split_streams(x):
    Bs, T, D = x.shape
    return x.reshape(Bs // S, S, T, D)

# %% Build per-layer per-stream metrics on post_mlp
entropy = np.zeros((L, S))                # nats
target_prob = np.zeros((L, S))            # P(' Paris')
top1_token = [["" for _ in range(S)] for _ in range(L)]
cosine = np.zeros((L, S, S))              # pair-wise residual cosine sim, last position
distinct_top1 = np.zeros(L, dtype=int)    # # of distinct top-1 tokens across streams

for i in range(L):
    streams = split_streams(captures[(i, "post_mlp")])  # (1, S, T, D)
    s_last = streams[0, :, -1, :]                       # (S, D)

    # cosine similarity between every pair
    s_norm = F.normalize(s_last, dim=-1)
    cosine[i] = (s_norm @ s_norm.T).cpu().numpy()

    top1s = []
    for s in range(S):
        x = streams[:, s]
        probs = lens_dist(x)
        ent = -(probs * (probs.clamp_min(1e-12)).log()).sum().item()
        entropy[i, s] = ent
        target_prob[i, s] = probs[target_id].item()
        top1_id = probs.argmax().item()
        top1_token[i][s] = tok.decode([top1_id])
        top1s.append(top1_id)
    distinct_top1[i] = len(set(top1s))

# %% Summary stats
print("\n=== per-layer summary ===")
print(f"{'L':>3} | {'entropy s0..s3':<35} | {'P(Paris)×1e3':<25} | {'distinct top1':>13} | mean cos(s_i, s_j)")
print("-" * 110)
for i in range(L):
    ent_s = "  ".join(f"{e:5.2f}" for e in entropy[i])
    p_s = "  ".join(f"{p*1000:6.2f}" for p in target_prob[i])
    # mean off-diagonal cosine
    off = (cosine[i].sum() - np.trace(cosine[i])) / (S*(S-1))
    print(f"{i:>3} | {ent_s:<35} | {p_s:<25} | {distinct_top1[i]:>13} | {off:+.3f}")

print(f"\nMax P(' Paris') across all layers/streams: "
      f"{target_prob.max():.4f} at layer {np.unravel_index(target_prob.argmax(), target_prob.shape)}")
print(f"Min entropy (most peaked) layer/stream: "
      f"{entropy.min():.2f} at {np.unravel_index(entropy.argmin(), entropy.shape)}")
print(f"Max entropy (most diffuse) layer/stream: "
      f"{entropy.max():.2f} at {np.unravel_index(entropy.argmax(), entropy.shape)}")

# Layer where streams first start disagreeing
first_disagree = next((i for i in range(L) if distinct_top1[i] > 1), None)
last_agree = next((i for i in range(L-1, -1, -1) if distinct_top1[i] == 1), None)
print(f"First layer streams disagree: {first_disagree}, last layer they all agree: {last_agree}")

# %% Visualizations
results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "logit_lens"
results_dir.mkdir(parents=True, exist_ok=True)

def save_heatmap(M, vmin, vmax, cmap, title, ylabel, xticks, xlabels, fname, cbar_label):
    fig, ax = plt.subplots(figsize=(max(4, len(xlabels)*0.8), 9))
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=15, fontsize=11)
    ax.set_xlabel("Stream", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([str(i) for i in range(M.shape[0])], fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            color = "white" if (vmax - v) < (vmax - vmin) * 0.4 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color=color)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(results_dir / fname, dpi=150, bbox_inches="tight")
    print(f"saved {results_dir / fname}")

# entropy heatmap — log10(V)≈10.8 nats max for V=50304
save_heatmap(
    entropy, vmin=0, vmax=np.log(50304), cmap="viridis",
    title="Per-layer × per-stream entropy of lens distribution (nats)",
    ylabel="Layer", xticks=range(S), xlabels=[f"s{i}" for i in range(S)],
    fname="fig4_entropy.png", cbar_label="entropy (nats)",
)

# target prob heatmap
save_heatmap(
    target_prob * 1000, vmin=0, vmax=max(1, target_prob.max() * 1000),
    cmap="YlOrRd",
    title=f"P({TARGET!r}) × 1000  per layer × stream (post_mlp)",
    ylabel="Layer", xticks=range(S), xlabels=[f"s{i}" for i in range(S)],
    fname="fig5_target_prob.png", cbar_label="P × 1000",
)

# pair-wise cosine sim averaged over layers (one figure: per-layer heatmap of average between streams)
mean_cos = cosine.mean(axis=0)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(mean_cos, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label="cosine")
ax.set_xticks(range(S)); ax.set_xticklabels([f"s{i}" for i in range(S)])
ax.set_yticks(range(S)); ax.set_yticklabels([f"s{i}" for i in range(S)])
for i in range(S):
    for j in range(S):
        ax.text(j, i, f"{mean_cos[i,j]:.2f}", ha="center", va="center", fontsize=10,
                color="white" if abs(mean_cos[i,j]) > 0.5 else "black")
ax.set_title("Mean cosine between streams (averaged over layers)")
plt.tight_layout()
fig.savefig(results_dir / "fig6_cos_sim.png", dpi=150, bbox_inches="tight")
print(f"saved {results_dir / 'fig6_cos_sim.png'}")

# distinct-top1 across layers
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(distinct_top1, marker="o")
ax.set_xlabel("Layer"); ax.set_ylabel("# distinct top-1 across 4 streams")
ax.set_title("Stream agreement vs layer (1 = all streams agree, 4 = all disagree)")
ax.set_yticks([1, 2, 3, 4])
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(results_dir / "fig7_distinct_top1.png", dpi=150, bbox_inches="tight")
print(f"saved {results_dir / 'fig7_distinct_top1.png'}")
# %%
