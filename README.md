# mhc-interp

Mechanistic interpretability comparison across the three [Realmbird/mhc-model-diff](https://huggingface.co/collections/Realmbird/mhc-model-diff) checkpoints — a 3-way ablation of hyper-connection schemes, all GPT-2-large shape (36L × 20H × 1280D) trained on a Dolma-v1_7 subset:

| name | repo | hyper-conn type | streams |
|---|---|---|---|
| residual | [Realmbird/mhc-781m-residual](https://huggingface.co/Realmbird/mhc-781m-residual) | `none` | 1 |
| mhc | [Realmbird/mhc-781m-mhc](https://huggingface.co/Realmbird/mhc-781m-mhc) | `mhc` | 4 |
| mhc-lite | [Realmbird/mhc-781m-mhc-lite](https://huggingface.co/Realmbird/mhc-781m-mhc-lite) | `mhc_lite` | 4 |

The goal is to see which mechanistic structures (induction heads, copy-suppression heads, sparse features, cross-stream redundancy, …) emerge in each variant — and whether the multi-stream MHC variants concentrate or distribute these structures differently than the residual baseline.

## Pipelines

All pipelines share a corpus loader (`src/mhc_interp/_loader.py`) that streams `monology/pile-uncopyrighted` (Pile is a major Dolma component, so it's effectively in-distribution). Activations are captured from each model's `model.py` via forward hooks and saved into per-pipeline result folders.

| Pipeline | Script | Output | One-line summary |
|---|---|---|---|
| Logit lens | [`logit_lens.py`](src/mhc_interp/logit_lens.py) | `results/logit_lens/{model}/` | Layer-by-layer top-K predictions per branch / per stream / per stream-subset |
| Attention patterns | [`attention_patterns.py`](src/mhc_interp/attention_patterns.py) | `results/attention_patterns/{model}/` | (L, H, T, T) attention reconstructed from hooked Q/K (flash SDPA materializes nothing) |
| Head detectors | [`head_finder.py`](src/mhc_interp/head_finder.py) | `results/heads/{slug}/{model}/` | Five canonical head types: prev-token, induction, duplicate, successor, copy-suppression. Pattern-based for the first three; per-head ablation (zero c_proj input slice) for successor + copy-suppression. |
| Multi-role analysis | [`multi_role_analysis.py`](src/mhc_interp/multi_role_analysis.py), [`multi_role_spotlight.py`](src/mhc_interp/multi_role_spotlight.py), [`layer_distribution.py`](src/mhc_interp/layer_distribution.py) | `results/analysis/` | Heads ranked top-K across multiple detectors; layer-distribution boxplot |
| Stream AE | [`stream_ae.py`](src/mhc_interp/stream_ae.py) | `results/stream_ae/{model}/` | Linear + 1-hidden MLP autoencoder mapping `stream_i → stream_j` per layer for mhc / mhc-lite; compared against `hc_mlp` static/dynamic α and β norms |
| SAE | [`sae_train.py`](src/mhc_interp/sae_train.py) | `results/sae/residual_L{L:02d}/` | TopK SAEs (D=1280 → 10240, k=64) on residual model `branch_mlp` output at L0 / L9 / L18 / L35 (≈ 0%, 25%, 50%, ~100% depth). SAE weights (`checkpoint.pt`) are gitignored. |
| Dashboard | [`app.py`](src/mhc_interp/app.py) | live | Streamlit viewer for every pipeline above |

## Quickstart

```bash
# install deps (GPU build pulled from the pytorch CUDA index)
uv sync --index-strategy unsafe-best-match

# run any pipeline (all idempotent, cached on first download)
uv run python src/mhc_interp/logit_lens.py
uv run python src/mhc_interp/attention_patterns.py
uv run python src/mhc_interp/head_finder.py
uv run python src/mhc_interp/multi_role_analysis.py
uv run python src/mhc_interp/multi_role_spotlight.py
uv run python src/mhc_interp/layer_distribution.py
uv run python src/mhc_interp/stream_ae.py
uv run python src/mhc_interp/sae_train.py

# browse everything in one place
uv run streamlit run src/mhc_interp/app.py
# → forward port 8501 (VSCode "Ports" panel auto-forwards localhost:8501)
```

The `_loader.get_token_corpus` call caches the tokenized corpus to `~/.cache/mhc_interp_tokens/` so repeat runs skip the streaming step.

## Selected findings so far

- **Layer concentration** ([results/analysis/layer_distribution_boxplot.png](results/analysis/layer_distribution_boxplot.png)). MHC variants pull most circuit-head types ~10 layers earlier than the residual baseline:

  | detector | residual median | mhc median | mhc-lite median |
  |---|---|---|---|
  | prev-token | L15 | L6 | L7 |
  | induction | L21 | L9 | L9 |
  | successor | L17 | L7 | L8 |
  | copy-suppression | L14 | L6 | L7 |

- **Multi-role heads** ([results/analysis/multi_role/spotlight.png](results/analysis/multi_role/spotlight.png)). All three variants produce ~6-7 heads with ≥2 roles (top-10 in 2 detectors). Only **MHC** produces 3-role heads at this threshold:
  - `mhc` `L7h17` — induction (0.92) + copy-suppression (0.995)
  - `mhc` `L7h15` — duplicate + successor + copy-suppression (3-role)
  - `mhc` `L6h11` — prev-token + successor + copy-suppression (3-role)

- **Cross-stream redundancy** (`results/stream_ae/`). Linear AE R² between MHC streams grows monotonically with depth in mhc-lite (0.69 → 0.93). MHC follows the same trend through L20, then collapses to negative R² on L25-35 — one MHC stream actively diverges in the late blocks (open question whether this is a dynamic-α effect or a scale mismatch).

- **SAE on residual `branch_mlp`** (`results/sae/`). TopK SAEs (k=64 / D_sae=10240) reach ~0.92-0.99 explained variance at L0 / L18 / L35 with 5K Adam steps; L9 has ~0 variance (the MLP barely writes anything mid-depth), so its SAE is degenerate.

## Repository layout

```
src/mhc_interp/
  _loader.py              shared model + corpus loader
  logit_lens.py           per-branch / per-stream lens
  attention_patterns.py   per-head attention capture
  head_finder.py          5 circuit-head detectors
  multi_role_analysis.py  multi-role rank table + heatmap
  multi_role_spotlight.py spotlight figure (residual vs MHC)
  layer_distribution.py   per-detector layer boxplot
  stream_ae.py            cross-stream AE (linear + MLP)
  sae_train.py            TopK SAE on residual branch_mlp
  app.py                  Streamlit dashboard (5 tabs)
results/                  outputs (large weights gitignored)
```

## Citation / model sources

The three checkpoints and their training recipe (paper-faithful Manifold-Constrained Hyper-Connections vs the lite variant vs a plain residual baseline) are from the [Realmbird/mhc-model-diff](https://huggingface.co/collections/Realmbird/mhc-model-diff) collection. See each model's HF README for hyperparameters and validation loss.

The MHC method itself is from [Ablate and Rescue (Wang et al., 2025; arXiv:2603.14833)](https://arxiv.org/abs/2603.14833).
