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
| Logit lens (single prompt) | [`logit_lens.py`](src/mhc_interp/logit_lens.py) | `results/logit_lens/{model}/` | Layer-by-layer top-K predictions per branch / per stream / per stream-subset |
| Logit lens (Dolma mean) | [`logit_lens_mean.py`](src/mhc_interp/logit_lens_mean.py) | `results/logit_lens/aggregate/` | KL/top-1/entropy averaged over 50k Dolma next-token positions (`ln_f` per-stream then sum, then unembed) |
| Attention patterns | [`attention_patterns.py`](src/mhc_interp/attention_patterns.py) | `results/attention_patterns/{model}/` | (L, H, T, T) attention reconstructed from hooked Q/K (flash SDPA materializes nothing) |
| Head detectors | [`head_finder.py`](src/mhc_interp/head_finder.py) | `results/heads/{slug}/{model}/` | Five canonical head types: prev-token, induction, duplicate, successor, copy-suppression. Pattern + ablation for prev-token / induction / duplicate; ablation-only for successor + copy-suppression. |
| Pattern vs ablation match | [`pattern_ablation_match.py`](src/mhc_interp/pattern_ablation_match.py) | `results/analysis/pattern_vs_ablation_match.{png,csv}` | 720-head scatter + Pearson r of pattern-score vs ablation ΔNLL per (detector, model). PT mhc r=+0.015. |
| Head population pattern | [`head_pattern_population.py`](src/mhc_interp/head_pattern_population.py), [`head_pattern_compare.py`](src/mhc_interp/head_pattern_compare.py), [`head_pattern_writeup_figure.py`](src/mhc_interp/head_pattern_writeup_figure.py) | `results/analysis/head_pattern_*` | Top-5 thumbnails + 720-head canonical-stripe-mass histogram |
| Path patching (prev-token) | [`path_patch_pt.py`](src/mhc_interp/path_patch_pt.py) | `results/heads/prev_token/{model}/path_patch.csv`, `results/analysis/prev_token_path_patch.png` | Direct/indirect ΔNLL decomposition. Direct = ablate head + freeze downstream blocks at baseline; total = standard ablation; indirect = total - direct. |
| Consumer attribution | [`path_patch_consumers.py`](src/mhc_interp/path_patch_consumers.py) | `results/heads/prev_token/{model}/consumer_attribution.npz`, `results/analysis/path_patch_consumers_{model}.png` | Per-candidate Frobenius norm of downstream attention shifts after ablation — finds heads that read from the candidate. |
| Verticality / sink analysis | [`verticality_figure.py`](src/mhc_interp/verticality_figure.py), [`indirect_sink_figure.py`](src/mhc_interp/indirect_sink_figure.py) | `results/analysis/verticality_pt_heads.png`, `indirect_pt_sink.png` | Average attention of pattern-top vs indirect-top PT heads + per-head column-mean dots. Verticality = max(col-mean) ÷ mean(col-mean). |
| Output distribution comparison | [`output_distribution_compare.py`](src/mhc_interp/output_distribution_compare.py) | `results/logit_lens/aggregate/output_distribution_compare.{png,csv,json}` | KL / JS / TV between models at last position over Dolma sample |
| Cat-sat token ranges | [`cat_sat_token_ranges.py`](src/mhc_interp/cat_sat_token_ranges.py) | `results/logit_lens/aggregate/cat_sat_token_layer_ranges.png`, `cat_sat_final_top3.png` | Per-token layer ranges for the "cat sat on the" prompt (tokens that hold top-1 ≥2 layers in any model) |
| Multi-role / layer dist | [`multi_role_analysis.py`](src/mhc_interp/multi_role_analysis.py), [`multi_role_spotlight.py`](src/mhc_interp/multi_role_spotlight.py), [`layer_distribution.py`](src/mhc_interp/layer_distribution.py) | `results/analysis/` | Heads ranked top-K across multiple detectors; per-detector layer boxplot |
| Stream AE | [`stream_ae.py`](src/mhc_interp/stream_ae.py) | `results/stream_ae/{model}/` | Linear + 1-hidden MLP autoencoder mapping `stream_i → stream_j` per layer for mhc / mhc-lite; compared against `hc_mlp` static/dynamic α and β norms |
| SAE | [`sae_train.py`](src/mhc_interp/sae_train.py) | `results/sae/residual_L{L:02d}/` | TopK SAEs (D=1280 → 10240, k=64) on residual model `branch_mlp` output at L0 / L9 / L18 / L35 (≈ 0%, 25%, 50%, ~100% depth). SAE weights (`checkpoint.pt`) are gitignored. |
| Dashboard | [`app.py`](src/mhc_interp/app.py) | live | Streamlit viewer for every pipeline above |

## Head detection — step by step

This is the methodology used by `head_finder.py`, `pattern_ablation_match.py`, `path_patch_pt.py`, and the verticality figures. Same pipeline applied to all 720 heads (36L × 20H) per model.

### Step 1 — Pick a probe (prompt + target)

Each head type has a hand-crafted prompt where the relevant kind of information is the only way to make the prediction. *"Probe" here means a probing prompt + ablation measurement, not a trained linear probe.*

| Detector | Prompt | Why it forces the mechanism |
|---|---|---|
| prev-token | `"When Mary and John went to the store, John gave a drink to Mary"` | Repeated names — model can only resolve via positional/prev-token info |
| induction | `[EOT] + R + R` where `R` = 25 random token IDs | Random repeats — only way to predict copy 2 is look back at copy 1 |
| duplicate | same as induction | Reuses the random-repeat structure |
| successor | 3 prompts averaged: days (`"... Friday"` → `" Saturday"`), numbers (`"... five"` → `" six"`), letters (`"... E"` → `" F"`) | Forces "increment by one" |
| copy-suppression | `"When John and Mary went to the store, John gave a drink to"`, target=`" Mary"`, distractor=`" John"` | Tests whether ablation flips prediction *toward* the duplicated name |

### Step 2 — Pattern detector (cheap, morphology only)

Capture (L, H, T, T) attention by hooking `c_attn` and reconstructing `softmax(QKᵀ/√d + causal)`. For each head, compute a stripe-mass score:

| Detector | Score formula |
|---|---|
| prev-token | `mean A[i, i-1]` over i ≥ 1 |
| induction | `mean A[1+n+i, 1+i+1]` over i in second copy |
| duplicate | `mean A[1+n+i, 1+i]` |

This measures **shape**, not function. A head with high stripe-mass might not actually do that computation.

### Step 3 — Ablation detector (causal, behavioral)

Per-head zero-ablation via `forward_pre_hook` on `block.branch_attn[1].c_proj` that zeros the slice `[:, :, h*hs:(h+1)*hs]` of the (B, T, 1280) input. One hook = full ablation (the hyper-conn α matrix runs *before* `branch_attn`, so c_proj is called once per layer per forward).

For each head `(l, h)`:
```
NLL_baseline = mean over target positions of -log P_clean(target)
NLL_ablated  = mean over target positions of -log P_ablated(target)
ΔNLL(l, h)   = NLL_ablated - NLL_baseline
```

Positive ΔNLL = head was useful. (Note: residual baseline skips ablation by default; pattern-only is fine for it. mhc / mhc_lite always run both — see `ABLATION_FOR_PATTERN_MODELS` in `head_finder.py`.)

### Step 4 — Combined criterion

`combined_pct = min(pattern_pct_rank, ablation_pct_rank)` — both detectors must agree for a head to rank well. This is what populates the "top-K" lists in `top_heads.json`. `top_heads_pattern_only.json` is the pattern-only ranking for comparison.

### Step 5 — Pattern vs ablation agreement (`pattern_ablation_match.py`)

Plot all 720 heads as one point in (pattern_score, ΔNLL) space per (detector, model). Compute Pearson r and `top-K ∩ top-K` overlap.

| Detector × Model | Pearson r | top-10 ∩ top-10 |
|---|---|---|
| Induction · mhc | +0.456 | 6 |
| Induction · mhc_lite | +0.239 | 3 |
| Prev-token · mhc | **+0.015** | **0** |
| Prev-token · mhc_lite | +0.107 | 1 |

→ **For prev-token in MHC, pattern shape and ablation effect are completely uncorrelated.** This is the headline finding.

### Step 6 — Path patching (direct/indirect, `path_patch_pt.py`)

For heads that score high on *ablation* but low on *pattern*, what's the head actually doing? Three forward passes per head:

| Pass | Setup | Output |
|---|---|---|
| Baseline | nothing ablated | `NLL_baseline` |
| Total | ablate `(l, h)` only | `NLL_total` |
| Direct | ablate `(l, h)` AND freeze every block at layer ≥ l+1 to its baseline output (`forward_post_hook` returns cached output) | `NLL_direct` |

```
total_effect    = NLL_total   - NLL_baseline
direct_effect   = NLL_direct  - NLL_baseline
indirect_effect = total_effect - direct_effect
```

`direct = 0` → head writes nothing useful directly to `lm_head`; its entire effect routes through downstream consumer heads.

→ **Every load-bearing PT head in mhc / mhc_lite has `direct = 0.000`.** Pure indirect routing.

### Step 7 — What shape are the load-bearing heads? (`indirect_sink_figure.py`)

For top-5 heads ranked by indirect ΔNLL, plot attention thumbnails + per-head column-mean. Compute verticality = max(col-mean) ÷ mean(col-mean); >2 = sink-like.

| Group | Verticality (avg of top-5) |
|---|---|
| Pattern-top, mhc | 2.07 |
| **Indirect-top, mhc** | **4.19** |
| Pattern-top, mhc_lite | 1.97 |
| **Indirect-top, mhc_lite** | **3.95** (one outlier: L7h14 is a textbook PT-stripe head with diag=0.997) |

→ Load-bearing PT heads in mhc are attention sinks. mhc_lite is mostly sinks with one canonical PT stripe head.

### Step 8 — Consumer attribution (`path_patch_consumers.py`)

For each top-3 candidate by indirect ΔNLL: ablate, capture all attention via hooks, compute `delta_norm[L, H] = ||baseline - ablated||_F`. Identifies which downstream heads' attention patterns shift when the candidate is ablated → the heads that *read* from the candidate.

### Outputs

```
results/heads/{slug}/{model}/
  attention.npy                  (L, H, T, T) attention, fp16 (shared across pattern detectors)
  tokens.json                    Token-id mirror of attention positions
  scores.csv                     per-head: layer, head, pattern_score, ablation_delta_nll, combined_pct
  top_heads.json                 top-K by combined criterion
  top_heads_pattern_only.json    top-K by pattern alone (for comparison)
  path_patch.csv                 prev-token only: layer, head, pattern_score, total_delta_nll, direct_delta_nll, indirect_delta_nll
  consumer_attribution.npz       prev-token only: per-candidate delta_norm (L, H) + ablated attention
results/analysis/
  pattern_vs_ablation_match.{png,csv}
  head_pattern_population_{prev_token,induction}.png
  prev_token_path_patch.png
  path_patch_consumers_{mhc,mhc_lite}.png
  verticality_pt_heads.png
  indirect_pt_sink.png
results/heads/summary.csv         all detectors × all models, top-K rows
```

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
  _loader.py                       shared model + corpus loader
  logit_lens.py                    per-branch / per-stream lens (single prompt)
  logit_lens_mean.py               KL/top-1/entropy averaged over Dolma sample
  attention_patterns.py            per-head attention capture
  head_finder.py                   5 circuit-head detectors (pattern + ablation)
  pattern_ablation_match.py        720-head pattern-vs-ablation Pearson r scatter
  head_pattern_population.py       canonical-stripe-mass histogram + top-5 thumbs
  head_pattern_compare.py          residual vs MHC top-K side-by-side
  head_pattern_writeup_figure.py   figure-ready combined panel
  path_patch_pt.py                 direct/indirect ΔNLL decomposition
  path_patch_consumers.py          downstream attention-shift attribution
  verticality_figure.py            pattern-top vs indirect-top avg attention
  indirect_sink_figure.py          standalone sink-shape evidence figure
  output_distribution_compare.py   KL/JS/TV between models at last position
  cat_sat_token_ranges.py          per-token layer ranges box-whisker
  multi_role_analysis.py           multi-role rank table + heatmap
  multi_role_spotlight.py          spotlight figure (residual vs MHC)
  layer_distribution.py            per-detector layer boxplot
  pattern_analysis.py              attention pattern heuristics
  top10_inventory.py               per-detector top-10 inventory
  stream_ae.py                     cross-stream AE (linear + MLP)
  sae_train.py                     TopK SAE on residual branch_mlp
  app.py                           Streamlit dashboard
results/                           outputs (large weights gitignored)
```

## Citation / model sources

The three checkpoints and their training recipe (paper-faithful Manifold-Constrained Hyper-Connections vs the lite variant vs a plain residual baseline) are from the [Realmbird/mhc-model-diff](https://huggingface.co/collections/Realmbird/mhc-model-diff) collection. See each model's HF README for hyperparameters and validation loss.

The MHC method itself is from [Ablate and Rescue (Wang et al., 2025; arXiv:2603.14833)](https://arxiv.org/abs/2603.14833).
