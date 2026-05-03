"""Streamlit dashboard for browsing the three Realmbird/mhc-781m-* checkpoints.

Run from the project root:
    uv run streamlit run src/mhc_interp/app.py

In remote VSCode, the "Ports" panel auto-forwards localhost:8501 → your laptop.
You can also `Cmd+Shift+P` → "Forward a Port" → 8501 if it doesn't pick up.

Tabs:
  * Attention Patterns — pick (layer, head), compare residual / mhc / mhc_lite
                        side-by-side. Download CSV for the current selection.
  * Logit Lens — pick model, browse heatmap + top-K tokens per (layer, view).
  * Head Inventory — for each canonical head type (prev-token, induction,
                     duplicate, successor, copy-suppression), browse the top-K
                     ranked heads per model with attention thumbnails.

Inputs are read from results/{attention_patterns,logit_lens,heads}/.
Generate with src/mhc_interp/{attention_patterns,logit_lens,head_finder}.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ATTN_ROOT = REPO_ROOT / "results" / "attention_patterns"
LENS_ROOT = REPO_ROOT / "results" / "logit_lens"
HEADS_ROOT = REPO_ROOT / "results" / "heads"
STREAM_AE_ROOT = REPO_ROOT / "results" / "stream_ae"
SAE_ROOT = REPO_ROOT / "results" / "sae"
MODELS = ["residual", "mhc", "mhc_lite"]
STREAM_MODELS = ["mhc", "mhc_lite"]  # only these have S>1
HEAD_TYPES = [
    ("prev_token", "Previous-token", "attention pattern: A[i, i-1] averaged."),
    ("induction", "Induction", "repeated random sequence: attend to next-after-previous."),
    ("duplicate", "Duplicate-token", "repeated random sequence: attend back to first occurrence."),
    ("successor", "Successor", "ablation Δ logP across days/numbers/letters; positive ⇒ supports successor."),
    ("copy_suppression", "Copy-suppression", "ablation Δ logP(' John') on IOI prompt; positive ⇒ head was suppressing."),
]

st.set_page_config(page_title="MHC interp", layout="wide")
st.title("MHC-781M interpretability dashboard")
st.caption("residual / mhc / mhc-lite — https://huggingface.co/collections/Realmbird/mhc-model-diff")


# ---------- loaders (cached so reruns are fast) ----------
@st.cache_data
def load_attn(name: str):
    d = ATTN_ROOT / name
    if not (d / "attention.npy").exists():
        return None
    A = np.load(d / "attention.npy")  # (L, H, T, T) float16
    meta = json.loads((d / "tokens.json").read_text())
    return A, meta


@st.cache_data
def load_lens(name: str):
    d = LENS_ROOT / name
    if not (d / "lens_data.npz").exists():
        return None
    npz = np.load(d / "lens_data.npz")
    meta = json.loads((d / "lens_meta.json").read_text())
    return {k: npz[k] for k in npz.files}, meta


@st.cache_data
def load_head_probe(slug: str, name: str):
    d = HEADS_ROOT / slug / name
    if not (d / "top_heads.json").exists():
        return None
    A = np.load(d / "attention.npy")
    meta = json.loads((d / "tokens.json").read_text())
    top = json.loads((d / "top_heads.json").read_text())
    return A, meta, top


@st.cache_data
def load_head_prompt(slug: str):
    p = HEADS_ROOT / slug / "prompt.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---------- attention pattern view ----------
def render_attention():
    st.header("Attention patterns")

    loaded = {n: load_attn(n) for n in MODELS}
    available = [n for n, v in loaded.items() if v is not None]
    if not available:
        st.warning("No attention data found. Run `uv run python src/mhc_interp/attention_patterns.py` first.")
        return

    sample = loaded[available[0]]
    _, sample_meta = sample
    L = sample_meta["shape"]["L"]
    H = sample_meta["shape"]["H"]
    tokens = sample_meta["tokens"]
    prompt = sample_meta["prompt"]

    st.markdown(f"**Prompt:** `{prompt}`  →  tokens: " + " · ".join(f"`{t}`" for t in tokens))

    c1, c2 = st.columns(2)
    with c1:
        layer = st.slider("Layer", 0, L - 1, 0, key="attn_layer")
    with c2:
        head = st.slider("Head", 0, H - 1, 0, key="attn_head")

    log_scale = st.checkbox("log color scale", value=False)
    show_values = st.checkbox("show weights in cells", value=True)

    cols = st.columns(len(available))
    for col, name in zip(cols, available):
        A, meta = loaded[name]
        pattern = A[layer, head].astype(np.float32)  # (T, T)
        with col:
            st.subheader(name)
            st.caption(f"`{meta['hyper_conn_type']}`  S={meta['hyper_conn_n']}  layer={layer}  head={head}")
            fig, ax = plt.subplots(figsize=(3.5, 3.2))
            mat = np.log10(pattern + 1e-9) if log_scale else pattern
            im = ax.imshow(
                mat, cmap="viridis", aspect="equal",
                vmin=-9 if log_scale else 0,
                vmax=0 if log_scale else 1,
            )
            T = pattern.shape[0]
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(tokens, fontsize=8)
            ax.set_xlabel("attended-to (key)", fontsize=9)
            ax.set_ylabel("from (query)", fontsize=9)
            if show_values:
                for i in range(T):
                    for j in range(T):
                        v = pattern[i, j]
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=7, color="white" if v > 0.5 else "black")
            fig.colorbar(im, ax=ax, fraction=0.046)
            st.pyplot(fig, clear_figure=True)

            df = pd.DataFrame(pattern, index=tokens, columns=tokens)
            df.index.name = "from \\ to"
            with st.expander("table"):
                st.dataframe(df.style.format("{:.4f}"), width="stretch")
            st.download_button(
                f"download CSV ({name}, layer {layer}, head {head})",
                data=df.to_csv().encode(),
                file_name=f"{name}_L{layer:02d}_H{head:02d}.csv",
                mime="text/csv",
                key=f"dl_{name}_{layer}_{head}",
            )

    with st.expander("global stats — head-by-head (current layer)"):
        rows = []
        for name in available:
            A, _ = loaded[name]
            for h in range(H):
                P = A[layer, h].astype(np.float32)
                T = P.shape[0]
                # entropy of attention rows averaged over query positions
                ent = -(P * np.log(P.clip(1e-12))).sum(axis=-1).mean()
                # diagonal mass: how much each token attends to itself
                diag = float(np.diag(P).mean())
                # last-row "previous-token" mass (j == i-1)
                prev = float(np.mean([P[i, i - 1] for i in range(1, T)]))
                rows.append({"model": name, "head": h,
                             "mean_row_entropy_nats": ent,
                             "mean_self_attn": diag,
                             "mean_prev_token_attn": prev})
        st.dataframe(pd.DataFrame(rows), width="stretch")


# ---------- logit lens view ----------
def render_lens():
    st.header("Logit lens")

    loaded = {n: load_lens(n) for n in MODELS}
    available = [n for n, v in loaded.items() if v is not None]
    if not available:
        st.warning("No lens data found. Run `uv run python src/mhc_interp/logit_lens.py` first.")
        return

    name = st.selectbox("Model", available)
    data, meta = loaded[name]
    figs = list(meta["figures"].keys())
    fig_key = st.selectbox(
        "Figure",
        figs,
        format_func=lambda k: {
            "fig1": "fig1 — branches + per-stream",
            "fig2": "fig2 — post-MLP stream subsets",
            "fig3": "fig3 — post-ATTN stream subsets",
        }.get(k, k),
    )

    probs = data[f"{fig_key}_probs"]  # (L, C)
    topk_ids = data[f"{fig_key}_topk_ids"]  # (L, C, K)
    topk_probs = data[f"{fig_key}_topk_probs"]  # (L, C, K)
    labels = meta["figures"][fig_key]["labels"]
    top1_tokens = meta["figures"][fig_key]["top1_tokens"]
    topk_tokens = meta["topk_tokens"][fig_key]
    L_, C = probs.shape

    st.caption(f"`{meta['repo_id']}`  ·  type=`{meta['hyper_conn_type']}`  S={meta['S']}  L={meta['L']}  prompt=`{meta['prompt']}`")

    fig, ax = plt.subplots(figsize=(max(8, C * 0.9), max(8, L_ * 0.28)))
    im = ax.imshow(probs, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="top-1 probability")
    for i in range(L_):
        for j in range(C):
            t = top1_tokens[i][j]
            disp = repr(t)[1:-1]
            if len(disp) > 10:
                disp = disp[:9] + "…"
            ax.text(j, i, disp, ha="center", va="center",
                    color="white" if probs[i, j] > 0.5 else "black", fontsize=7)
    ax.set_xticks(range(C))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(L_))
    ax.set_yticklabels([str(i) for i in range(L_)], fontsize=7)
    ax.set_xlabel("View")
    ax.set_ylabel("Layer")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Drill-down")
    c1, c2 = st.columns(2)
    with c1:
        layer = st.slider("Layer", 0, L_ - 1, L_ - 1, key="lens_layer")
    with c2:
        view_idx = st.selectbox("View", range(C), format_func=lambda i: labels[i], key="lens_view")
    K = topk_probs.shape[2]
    st.write(f"Top-{K} tokens at layer {layer}, view `{labels[view_idx]}`:")
    df = pd.DataFrame({
        "rank": range(1, K + 1),
        "token_id": topk_ids[layer, view_idx].tolist(),
        "token": [topk_tokens[layer][view_idx][k] for k in range(K)],
        "prob": topk_probs[layer, view_idx].tolist(),
    })
    st.dataframe(df, width="stretch")


# ---------- head inventory view ----------
def _render_head_thumbnail(A_lh: np.ndarray, tokens: list[str], title: str):
    """Small attention heatmap; A_lh is (T, T)."""
    T = A_lh.shape[0]
    short = [t.strip() or t for t in tokens]
    if T > 14:  # truncate long token labels for readability on the synthetic probe
        short = [s[:6] for s in short]
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.imshow(A_lh.astype(np.float32), cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_title(title, fontsize=8, pad=2)
    if T <= 14:
        ax.set_xticks(range(T)); ax.set_yticks(range(T))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=5)
        ax.set_yticklabels(short, fontsize=5)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(length=0, pad=1)
    return fig


def render_heads():
    st.header("Head inventory")

    summary_path = HEADS_ROOT / "summary.csv"
    if not summary_path.exists():
        st.warning("No head data found. Run `uv run python src/mhc_interp/head_finder.py` first.")
        return

    summary = pd.read_csv(summary_path)
    st.caption(f"Loaded summary.csv ({len(summary)} rows). One probe per head type, scored across L=36 × H=20 per model.")

    type_labels = [f"{label} — {desc}" for _, label, desc in HEAD_TYPES]
    idx = st.selectbox(
        "Head type",
        range(len(HEAD_TYPES)),
        format_func=lambda i: type_labels[i],
        key="head_type",
    )
    slug, label, desc = HEAD_TYPES[idx]

    prompt_meta = load_head_prompt(slug)
    if prompt_meta:
        if "probe" in prompt_meta and "text" in prompt_meta["probe"]:
            st.markdown(f"**Probe:** `{prompt_meta['probe']['text']}`")
        elif "probes" in prompt_meta:
            st.markdown("**Probes (averaged):**")
            for p in prompt_meta["probes"]:
                if "text" in p:
                    target = p.get("target", "")
                    st.markdown(f"- `{p['text']}` → `{target}`")
        elif "probe" in prompt_meta and prompt_meta["probe"].get("prompt_kind") == "synthetic_repeat":
            n = prompt_meta["probe"]["n"]
            seed = prompt_meta["probe"]["seed"]
            st.markdown(f"**Probe:** `[EOT] + R + R` with `R` = {n} random ids (seed={seed}). Tokens may look like noise — that's the point.")

    K = st.slider("Top-K heads to show", 3, 20, 10, key="head_topk")

    cols = st.columns(len(MODELS))
    for col, name in zip(cols, MODELS):
        loaded = load_head_probe(slug, name)
        with col:
            st.subheader(name)
            if loaded is None:
                st.write("(no data)")
                continue
            A, meta, top = loaded
            tokens = meta["tokens"]
            for entry in top[:K]:
                li, hi, sc = entry["layer"], entry["head"], entry["score"]
                pattern = A[li, hi]
                title = f"L{li}  h{hi}  score={sc:+.3f}"
                fig = _render_head_thumbnail(pattern, tokens, title)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
                if slug == "copy_suppression":
                    st.caption(
                        f"Δ logP(distractor)={entry['score']:+.3f}  ·  "
                        f"Δ logP(target)={entry.get('delta_target', 0):+.3f}"
                    )

    with st.expander("full ranking table"):
        df = summary[summary["head_type"] == slug].sort_values(["model", "rank"])
        st.dataframe(df, width="stretch")
        st.download_button(
            f"download {slug} summary CSV",
            data=df.to_csv(index=False).encode(),
            file_name=f"head_inventory_{slug}.csv",
            mime="text/csv",
            key=f"dl_heads_{slug}",
        )


# ---------- stream AE view ----------
@st.cache_data
def load_stream_ae(model: str):
    d = STREAM_AE_ROOT / model
    if not (d / "r2_linear.npy").exists():
        return None
    return {
        "r2_linear": np.load(d / "r2_linear.npy"),
        "r2_mlp": np.load(d / "r2_mlp.npy"),
        "alpha_beta": pd.read_csv(d / "alpha_beta.csv"),
        "summary": pd.read_csv(d / "cross_stream_summary.csv"),
    }


def render_stream_ae():
    st.header("Stream autoencoders")
    st.caption(
        "AE that learns a transformation from one residual stream to another at the same layer "
        "(linear + 1-hidden-layer MLP). High R² ⇒ streams are highly redundant — the hyper-conn "
        "is mixing information heavily. Low R² ⇒ streams carry distinct content."
    )

    loaded = {m: load_stream_ae(m) for m in STREAM_MODELS}
    available = [m for m, v in loaded.items() if v is not None]
    if not available:
        st.warning("No stream-AE data found. Run `uv run python src/mhc_interp/stream_ae.py` first.")
        return

    name = st.selectbox("Model", available, key="stream_ae_model")
    data = loaded[name]
    r2_lin, r2_mlp = data["r2_linear"], data["r2_mlp"]
    L, S, _ = r2_lin.shape

    flavor = st.radio("AE flavor", ["linear", "MLP"], horizontal=True, key="stream_ae_flavor")
    M = r2_lin if flavor == "linear" else r2_mlp

    # Per-layer mean off-diagonal R²
    off = [(i, j) for i in range(S) for j in range(S) if i != j]
    means = np.array([np.mean([M[li, i, j] for i, j in off]) for li in range(L)])

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(range(L), means, marker="o", color="#c0392b", linewidth=1.5)
    ax.set_xlabel("layer"); ax.set_ylabel(f"R² ({flavor} AE, mean off-diag)")
    ax.set_ylim(min(-0.1, float(means.min()) - 0.05), 1.05)
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(alpha=0.3)
    ax.set_title(f"{name}: cross-stream redundancy by layer")
    st.pyplot(fig, clear_figure=True); plt.close(fig)

    layer = st.slider("Layer (for the per-pair matrix)", 0, L - 1, L // 2, key="stream_ae_layer")
    st.write(f"**Layer {layer}** — R²(src → dst) for each (src, dst) stream pair:")
    fig2, ax2 = plt.subplots(figsize=(4.2, 3.6))
    M_layer = M[layer]
    im = ax2.imshow(M_layer, cmap="RdYlGn", vmin=-0.5, vmax=1.0, aspect="equal")
    for i in range(S):
        for j in range(S):
            v = M_layer[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="black" if abs(v) < 0.5 else "white", fontsize=10)
    ax2.set_xlabel("dst stream"); ax2.set_ylabel("src stream")
    ax2.set_xticks(range(S)); ax2.set_xticklabels([f"s{i}" for i in range(S)])
    ax2.set_yticks(range(S)); ax2.set_yticklabels([f"s{i}" for i in range(S)])
    fig2.colorbar(im, ax=ax2, fraction=0.046)
    st.pyplot(fig2, clear_figure=True); plt.close(fig2)

    with st.expander("hyper-conn parameter norms (hc_mlp)"):
        st.dataframe(data["alpha_beta"], width="stretch")
    with st.expander("per-layer R² summary CSV"):
        st.dataframe(data["summary"], width="stretch")
        st.download_button(
            f"download {name} stream_ae summary CSV",
            data=data["summary"].to_csv(index=False).encode(),
            file_name=f"stream_ae_{name}_summary.csv",
            mime="text/csv", key=f"dl_streamae_{name}",
        )


# ---------- SAE view ----------
@st.cache_data
def load_sae_layer(layer: int):
    d = SAE_ROOT / f"residual_L{layer:02d}"
    if not (d / "training_log.csv").exists():
        return None
    log = pd.read_csv(d / "training_log.csv")
    stats = json.loads((d / "stats.json").read_text())
    return log, stats


def render_sae():
    st.header("Sparse autoencoders (residual model)")
    st.caption(
        "TopK SAEs trained on the residual model's `branch_mlp` output (the per-layer "
        "MLP delta written into the residual stream). Sparse features capture what the "
        "MLP is writing at each depth."
    )

    layers = sorted(int(p.name.replace("residual_L", "")) for p in SAE_ROOT.iterdir()
                    if p.is_dir() and p.name.startswith("residual_L"))
    if not layers:
        st.warning("No SAE data found. Run `uv run python src/mhc_interp/sae_train.py` first.")
        return

    # Summary table across layers
    rows = []
    for li in layers:
        loaded = load_sae_layer(li)
        if loaded is None:
            continue
        log, stats = loaded
        rows.append({
            "layer": li, "d_sae": stats["d_sae"], "k": stats["topk_k"],
            "n_tokens": stats["n_tokens"],
            "held_mse": stats["held_mse"], "L0": stats["l0"],
            "explained_variance": stats["explained_variance"],
            "dead_pct": stats["dead_pct"],
            "elapsed_sec": stats["elapsed_sec"],
        })
    st.subheader("Final stats per layer")
    st.dataframe(pd.DataFrame(rows), width="stretch")

    # Training curves
    metrics = [("held_mse", "Held-out MSE"), ("l0", "L0 (avg active)"),
               ("dead_pct", "Dead-feature fraction"), ("explained_variance", "Explained variance")]
    sel = st.selectbox("Metric", [t for _, t in metrics], key="sae_metric")
    key = next(k for k, t in metrics if t == sel)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    cmap = plt.cm.viridis
    for i, li in enumerate(layers):
        loaded = load_sae_layer(li)
        if loaded is None: continue
        log, _ = loaded
        ax.plot(log["iter"], log[key], color=cmap(i / max(1, len(layers) - 1)),
                label=f"L{li}", linewidth=1.5)
    ax.set_xlabel("iter"); ax.set_ylabel(sel)
    ax.grid(alpha=0.3); ax.legend(loc="best")
    ax.set_title(f"{sel} per training step")
    st.pyplot(fig, clear_figure=True); plt.close(fig)


# ---------- driver ----------
tab_attn, tab_lens, tab_heads, tab_stream_ae, tab_sae = st.tabs(
    ["Attention Patterns", "Logit Lens", "Head Inventory", "Stream AE", "SAE"]
)
with tab_attn:
    render_attention()
with tab_lens:
    render_lens()
with tab_heads:
    render_heads()
with tab_stream_ae:
    render_stream_ae()
with tab_sae:
    render_sae()
