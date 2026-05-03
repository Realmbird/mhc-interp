# %%
"""Capture attention patterns for every (layer, head) of the three
Realmbird/mhc-781m-* checkpoints on a shared prompt.

Models (https://huggingface.co/collections/Realmbird/mhc-model-diff):
  - Realmbird/mhc-781m-residual   (none, n=1)   → 36 layers × 20 heads = 720 maps
  - Realmbird/mhc-781m-mhc        (mhc,  n=4)   → 720 maps × 4 streams
  - Realmbird/mhc-781m-mhc-lite   (mhc_lite, n=4) → 720 maps × 4 streams

Each block uses flash SDPA (is_causal=True), which doesn't materialize the
attention weights. We hook `c_attn` to grab its Q/K projections, then
reconstruct A = softmax(QKᵀ/√d + causal_mask) ourselves.

Note: MHC / MHC-lite mix the S streams into a single (B, T, D) branch input
*before* attention runs (see hyper_conn `width_connection`), so attention is
computed once per layer regardless of S. We end up with one (H, T, T) per
layer for every variant.

Outputs (per model `name` ∈ {residual, mhc, mhc_lite}):
  results/attention_patterns/{name}/attention.npy        — float16 (L, H, T, T)
  results/attention_patterns/{name}/attention_long.csv   — long-form: layer,head,src,dst,weight
  results/attention_patterns/{name}/tokens.json          — prompt + token strings + shape

Use the Streamlit dashboard to browse interactively:
    uv run streamlit run src/mhc_interp/app.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from mhc_interp._loader import attn_from_qkv, load_model_from_repo

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("gpt2")

# Same shared prompt as the logit-lens runs (all 3 top-1 ' floor').
PROMPT = "The cat sat on the"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = REPO_ROOT / "results" / "attention_patterns"

MODELS = [
    {"name": "residual", "repo_id": "Realmbird/mhc-781m-residual"},
    {"name": "mhc",      "repo_id": "Realmbird/mhc-781m-mhc"},
    {"name": "mhc_lite", "repo_id": "Realmbird/mhc-781m-mhc-lite"},
]


# %% Per-model run
def run_model(name: str, repo_id: str):
    print(f"\n{'=' * 80}\n>>> {name}  ({repo_id})\n{'=' * 80}")
    model, cfg = load_model_from_repo(repo_id, device)
    L, H, S = cfg.n_layer, cfg.n_head, cfg.hyper_conn_n
    print(f"L={L}  H={H}  S={S}  type={cfg.hyper_conn_type}")

    # Hook c_attn output (= qkv concatenated) on every layer's attention branch.
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_i):
        def hook(_m, _inp, out):
            captured[layer_i] = out.detach()
        return hook

    handles = []
    for i, block in enumerate(model.transformer.h):
        # branch_attn = Sequential(LayerNorm, CausalSelfAttention); CausalSelfAttention is index 1.
        attn_module = block.branch_attn[1]
        handles.append(attn_module.c_attn.register_forward_hook(make_hook(i)))

    ids = torch.tensor([tok.encode(PROMPT)], dtype=torch.long, device=device)
    token_strs = [tok.decode([t]) for t in ids[0].tolist()]
    T = ids.shape[1]
    print(f"prompt {PROMPT!r} -> {T} tokens: {token_strs}")

    with torch.no_grad():
        _ = model(ids)
    for h in handles:
        h.remove()

    # Build (L, H, T, T). Attention is computed once per layer regardless of S.
    all_attn = torch.zeros((L, H, T, T), dtype=torch.float16)
    for i in range(L):
        a = attn_from_qkv(captured[i], H)  # (1, H, T, T)
        assert a.shape[0] == 1, f"unexpected batch dim from c_attn: {a.shape}"
        all_attn[i] = a[0].to(torch.float16).cpu()

    out_dir = RESULTS_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "attention.npy", all_attn.numpy())
    print(f"saved {out_dir / 'attention.npy'}  shape={tuple(all_attn.shape)}")

    with open(out_dir / "tokens.json", "w") as f:
        json.dump({
            "model_name": name,
            "repo_id": repo_id,
            "hyper_conn_type": cfg.hyper_conn_type,
            "hyper_conn_n": S,
            "prompt": PROMPT,
            "tokens": token_strs,
            "token_ids": ids[0].tolist(),
            "shape": {"L": L, "H": H, "T": T},
            "tensor_dtype": "float16",
            "tensor_path": "attention.npy",
        }, f, indent=2)
    print(f"saved {out_dir / 'tokens.json'}")

    # Long-form CSV: one row per (layer, head, src, dst). 36*20*5*5 = 18000 rows.
    csv_path = out_dir / "attention_long.csv"
    attn_np = all_attn.float().numpy()
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "head", "src_idx", "src_token", "dst_idx", "dst_token", "weight"])
        for layer_i in range(L):
            for head_i in range(H):
                A = attn_np[layer_i, head_i]
                for src in range(T):
                    for dst in range(T):
                        w.writerow([
                            layer_i, head_i, src, token_strs[src],
                            dst, token_strs[dst], f"{float(A[src, dst]):.6f}",
                        ])
    print(f"saved {csv_path}  ({L*H*T*T} rows)")

    del model, captured, all_attn
    if device == "cuda":
        torch.cuda.empty_cache()


# %% Drive
if __name__ == "__main__":
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    for spec in MODELS:
        run_model(spec["name"], spec["repo_id"])

# %%
