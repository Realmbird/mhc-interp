# %%
"""Find canonical "circuit head" types in the three Realmbird/mhc-781m-* checkpoints.

Five probes, one folder per probe under results/heads/:
  - prev_token         (attention pattern)
  - induction          (attention pattern, on a repeated random sequence)
  - duplicate          (attention pattern, shares the induction probe's NPY)
  - successor          (per-head ablation, averaged across 3 ordinal probes)
  - copy_suppression   (per-head ablation on an IOI-style prompt)

For each (probe, model) we save:
  attention.npy        (L, H, T, T) fp16     — the captured attention
  tokens.json          prompt + token strings + ids + shape
  scores.csv           layer, head, score, [...optional ablation columns]
  top_heads.json       top 20 heads ranked by score

Plus a top-level summary.csv.

Run:
    uv run python src/mhc_interp/head_finder.py
"""
from __future__ import annotations

import csv
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from mhc_interp._loader import attn_from_qkv, load_model_from_repo

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("gpt2")
EOT_ID = 50256  # gpt2 <|endoftext|>, doubles as BOS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = REPO_ROOT / "results" / "heads"

MODELS = [
    {"name": "residual", "repo_id": "Realmbird/mhc-781m-residual"},
    {"name": "mhc",      "repo_id": "Realmbird/mhc-781m-mhc"},
    {"name": "mhc_lite", "repo_id": "Realmbird/mhc-781m-mhc-lite"},
]

# How many top heads to surface in top_heads.json + summary.csv.
TOP_K = 20

# Models for which we also run an ablation cross-check on the pattern detectors.
# Residual is excluded by user request — pattern is sufficient for the baseline.
ABLATION_FOR_PATTERN_MODELS = {"mhc", "mhc_lite"}


# %% ---------- Probe construction ----------
def _ids(text: str, prepend_eot: bool = False) -> list[int]:
    ids = tok.encode(text)
    if prepend_eot:
        ids = [EOT_ID] + ids
    return ids


def _decode(ids: Iterable[int]) -> list[str]:
    return [tok.decode([int(i)]) for i in ids]


def _make_induction_probe(n: int = 25, seed: int = 0) -> dict:
    """[EOT] + R + R, R = n random ids in a "safe" range."""
    rng = np.random.default_rng(seed)
    R = rng.integers(low=1000, high=40000, size=n).tolist()
    ids = [EOT_ID] + R + R
    return {
        "prompt_kind": "synthetic_repeat",
        "n": n,
        "seed": seed,
        "first_offset": 1,           # first occurrence of R starts at index 1
        "second_offset": 1 + n,      # second occurrence starts at index 1+n
        "ids": ids,
        "tokens": _decode(ids),
    }


def _make_text_probe(text: str, prepend_eot: bool = False) -> dict:
    ids = _ids(text, prepend_eot)
    return {
        "prompt_kind": "text",
        "text": text,
        "prepend_eot": prepend_eot,
        "ids": ids,
        "tokens": _decode(ids),
    }


def _ablation_probe(text: str, target: str, distractor: str | None = None) -> dict:
    """Text probe with a target token (and optional distractor for copy-suppression)."""
    p = _make_text_probe(text)
    p["target"] = target
    p["target_id"] = tok.encode(target)[0]
    if distractor is not None:
        p["distractor"] = distractor
        p["distractor_id"] = tok.encode(distractor)[0]
    return p


SUCCESSOR_PROBES = [
    _ablation_probe("Monday Tuesday Wednesday Thursday Friday", " Saturday"),
    _ablation_probe(" one two three four five", " six"),
    _ablation_probe("A B C D E", " F"),
]

PROBES: dict[str, dict] = {
    "prev_token": {
        "detector": "pattern",
        "probe": _make_text_probe(
            "When Mary and John went to the store, John gave a drink to Mary"
        ),
    },
    "induction": {
        "detector": "pattern",
        "probe": _make_induction_probe(n=25, seed=0),
        "shared_attention_with": "duplicate",
    },
    "duplicate": {
        "detector": "pattern",
        "probe": _make_induction_probe(n=25, seed=0),  # identical seed → same prompt
        "shared_attention_with": "induction",
    },
    "successor": {
        "detector": "ablation_multi",
        "probes": SUCCESSOR_PROBES,
        # Use the first probe's attention as the thumbnail.
        "probe": SUCCESSOR_PROBES[0],
    },
    "copy_suppression": {
        "detector": "ablation",
        "probe": _ablation_probe(
            "When John and Mary went to the store, John gave a drink to",
            target=" Mary",
            distractor=" John",
        ),
    },
}


# %% ---------- Attention capture ----------
def _register_capture_hooks(model, captured: dict[int, torch.Tensor]) -> list:
    handles = []
    for i, block in enumerate(model.transformer.h):
        c_attn = block.branch_attn[1].c_attn
        def make_hook(layer_i):
            def hook(_m, _inp, out):
                captured[layer_i] = out.detach()
            return hook
        handles.append(c_attn.register_forward_hook(make_hook(i)))
    return handles


@torch.no_grad()
def capture_attention(model, ids: list[int], n_head: int) -> np.ndarray:
    """Run one forward pass and return (L, H, T, T) attention as fp16 numpy."""
    captured: dict[int, torch.Tensor] = {}
    handles = _register_capture_hooks(model, captured)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    try:
        _ = model(x)
    finally:
        for h in handles:
            h.remove()
    L = len(model.transformer.h)
    T = x.shape[1]
    out = torch.zeros((L, n_head, T, T), dtype=torch.float16)
    for i in range(L):
        a = attn_from_qkv(captured[i], n_head)  # (1, H, T, T)
        assert a.shape[0] == 1, f"unexpected batch dim: {a.shape}"
        out[i] = a[0].to(torch.float16).cpu()
    return out.numpy()


# %% ---------- Pattern-based scorers ----------
def score_prev_token(A: np.ndarray) -> np.ndarray:
    """A: (L, H, T, T). Returns (L, H) = mean_{i≥1} A[..., i, i-1]."""
    L, H, T, _ = A.shape
    if T < 2:
        return np.zeros((L, H), dtype=np.float32)
    # Diagonal one-below: A[..., i, i-1] for i in 1..T-1
    rows = np.arange(1, T)
    return A[:, :, rows, rows - 1].mean(axis=-1).astype(np.float32)


def score_induction(A: np.ndarray, n: int, second_offset: int, first_offset: int) -> np.ndarray:
    """At pos (second_offset + i) attend to (first_offset + i + 1) — "next-after-previous"."""
    L, H, T, _ = A.shape
    out = np.zeros((L, H), dtype=np.float32)
    rows = []
    cols = []
    for i in range(n - 1):  # skip last i: no token after end of first repeat
        r = second_offset + i
        c = first_offset + i + 1
        if r < T and c < T:
            rows.append(r); cols.append(c)
    if rows:
        rows = np.array(rows); cols = np.array(cols)
        out = A[:, :, rows, cols].mean(axis=-1).astype(np.float32)
    return out


def score_duplicate(A: np.ndarray, n: int, second_offset: int, first_offset: int) -> np.ndarray:
    """At pos (second_offset + i) attend back to (first_offset + i) — same token, earlier."""
    L, H, T, _ = A.shape
    out = np.zeros((L, H), dtype=np.float32)
    rows, cols = [], []
    for i in range(n):
        r = second_offset + i
        c = first_offset + i
        if r < T and c < T:
            rows.append(r); cols.append(c)
    if rows:
        rows = np.array(rows); cols = np.array(cols)
        out = A[:, :, rows, cols].mean(axis=-1).astype(np.float32)
    return out


# %% ---------- Per-head ablation ----------
@contextmanager
def ablate_head(model, layer_i: int, head_i: int, hs: int):
    """forward_pre_hook on c_proj that zeros the slice [:, :, h*hs:(h+1)*hs].

    In all 3 variants, c_proj is called once per layer per forward pass — the
    hyper_conn wrappers mix streams BEFORE branch_attn runs. So this hook is
    a single point of effect for ablating one head's contribution.
    """
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
def _last_pos_logp(model, ids: list[int]) -> torch.Tensor:
    """Return log-softmax over vocab at the last position. Shape (V,)."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x)
    return F.log_softmax(logits[0, -1], dim=-1)


@torch.no_grad()
def ablation_grid(
    model,
    cfg,
    ids: list[int],
    target_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (baseline (|targets|,), ablated (L, H, |targets|)) — log-probs.

    Runs one baseline forward + L*H ablated forwards.
    """
    L, H = cfg.n_layer, cfg.n_head
    hs = cfg.n_embd // H
    baseline_lp = _last_pos_logp(model, ids)
    baseline_t = baseline_lp[target_ids].float().cpu().numpy()  # (|targets|,)

    out = np.zeros((L, H, len(target_ids)), dtype=np.float32)
    for li in range(L):
        for hi in range(H):
            with ablate_head(model, li, hi, hs):
                lp = _last_pos_logp(model, ids)
            out[li, hi] = lp[target_ids].float().cpu().numpy()
    return baseline_t, out


@torch.no_grad()
def _nll_at_positions(model, ids: list[int], positions: list[int] | None) -> float:
    """Cross-entropy of next-token prediction at the given positions (or all
    positions when `positions` is None). Returns mean nats.

    `positions` are query positions i — predictions at i target token ids[i+1].
    """
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x, torch.zeros_like(x))  # `targets=...` makes the model
                                                  # return per-position logits
    log_p = F.log_softmax(logits, dim=-1)         # (1, T, V)
    targets = x[:, 1:]                            # (1, T-1)
    log_p_pred = log_p[:, :-1, :]                 # (1, T-1, V)
    nll = -log_p_pred.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    if positions is not None:
        valid = [p for p in positions if 0 <= p < nll.shape[1]]
        nll = nll[:, valid]
    return float(nll.mean().item())


@torch.no_grad()
def ablation_nll_grid(
    model,
    cfg,
    ids: list[int],
    positions: list[int] | None = None,
) -> tuple[float, np.ndarray]:
    """For each (L, H) ablate the head and compute NLL at the given positions.

    Returns (baseline_nll, ablated_nll[L, H]). Δ = ablated - baseline > 0
    means the head's contribution lowered NLL ⇒ head was useful.
    """
    L, H = cfg.n_layer, cfg.n_head
    hs = cfg.n_embd // H
    baseline = _nll_at_positions(model, ids, positions)
    out = np.zeros((L, H), dtype=np.float32)
    for li in range(L):
        for hi in range(H):
            with ablate_head(model, li, hi, hs):
                out[li, hi] = _nll_at_positions(model, ids, positions)
    return baseline, out


# %% ---------- Output writers ----------
def _write_top_heads(scores: np.ndarray, out_path: Path, k: int = TOP_K, **extra_cols) -> list[dict]:
    """scores: (L, H). Writes top_heads.json with the K heads of largest score."""
    L, H = scores.shape
    flat = [(li, hi, float(scores[li, hi])) for li in range(L) for hi in range(H)]
    flat.sort(key=lambda r: -r[2])
    top = []
    for rank, (li, hi, s) in enumerate(flat[:k], start=1):
        row = {"rank": rank, "layer": li, "head": hi, "score": s}
        for col, arr in extra_cols.items():
            row[col] = float(arr[li, hi])
        top.append(row)
    out_path.write_text(json.dumps(top, indent=2))
    return top


def _write_scores_csv(out_path: Path, scores: np.ndarray, extra_cols: dict[str, np.ndarray] | None = None):
    L, H = scores.shape
    extra_cols = extra_cols or {}
    cols = ["layer", "head", "score"] + list(extra_cols.keys())
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for li in range(L):
            for hi in range(H):
                row = [li, hi, f"{float(scores[li, hi]):.6f}"]
                for c in extra_cols:
                    row.append(f"{float(extra_cols[c][li, hi]):.6f}")
                w.writerow(row)


def _write_tokens_json(out_dir: Path, name: str, repo_id: str, cfg, probe: dict, T: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": name,
        "repo_id": repo_id,
        "hyper_conn_type": cfg.hyper_conn_type,
        "hyper_conn_n": cfg.hyper_conn_n,
        "tokens": probe["tokens"],
        "token_ids": probe["ids"],
        "prompt_kind": probe.get("prompt_kind"),
        "shape": {"L": cfg.n_layer, "H": cfg.n_head, "T": T},
        "tensor_dtype": "float16",
        "tensor_path": "attention.npy",
    }
    if "text" in probe:
        payload["prompt"] = probe["text"]
    (out_dir / "tokens.json").write_text(json.dumps(payload, indent=2))


def _write_prompt_json(slug_dir: Path, slug: str, spec: dict):
    """Writes results/heads/{slug}/prompt.json (probe metadata, not per-model)."""
    slug_dir.mkdir(parents=True, exist_ok=True)
    payload = {"slug": slug, "detector": spec["detector"]}
    if "shared_attention_with" in spec:
        payload["shared_attention_with"] = spec["shared_attention_with"]
    if spec["detector"] == "ablation_multi":
        payload["probes"] = [
            {k: v for k, v in p.items() if k != "ids" or True}  # keep ids
            for p in spec["probes"]
        ]
    else:
        payload["probe"] = spec["probe"]
    (slug_dir / "prompt.json").write_text(json.dumps(payload, indent=2))


# %% ---------- Per-(model, probe) runners ----------
def run_pattern_probe(slug: str, spec: dict, model, cfg, name: str, repo_id: str,
                       attn_cache: dict[tuple[str, str], np.ndarray]) -> tuple[np.ndarray, list[dict]]:
    """Pattern-based detector. Returns (scores, top_heads)."""
    probe = spec["probe"]
    T = len(probe["ids"])
    out_dir = RESULTS_ROOT / slug / name

    # Capture attention (or reuse if shared).
    cache_key = (name, json.dumps(probe["ids"]))  # ids define the prompt uniquely
    if cache_key in attn_cache:
        A = attn_cache[cache_key]
    else:
        A = capture_attention(model, probe["ids"], cfg.n_head)
        attn_cache[cache_key] = A

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "attention.npy", A)
    _write_tokens_json(out_dir, name, repo_id, cfg, probe, T)

    # Score.
    if slug == "prev_token":
        scores = score_prev_token(A)
    elif slug == "induction":
        n = probe["n"]
        scores = score_induction(A, n, probe["second_offset"], probe["first_offset"])
    elif slug == "duplicate":
        n = probe["n"]
        scores = score_duplicate(A, n, probe["second_offset"], probe["first_offset"])
    else:
        raise ValueError(f"unknown pattern probe: {slug}")

    extras: dict[str, np.ndarray] = {}
    combined_score = None
    if name in ABLATION_FOR_PATTERN_MODELS:
        # Pick the ablation task that matches this detector's mechanism:
        # prev-token: NLL on the natural-text probe (general LM-loss signal).
        # induction:  NLL at second-half positions of [EOT]+R+R (Olsson §3.4).
        # duplicate:  same probe as induction (DT and IH share the underlying
        #             repeated-sequence task; pattern is what separates them).
        if slug == "prev_token":
            ablation_ids = probe["ids"]
            positions = None  # average over all next-token positions
        else:
            # synthetic [EOT]+R+R built earlier; positions to score are the
            # second-half query positions (i.e. predicting tokens at
            # second_offset+1 .. T-1 from the residual at second_offset .. T-2).
            n = probe["n"]
            second_off = probe["second_offset"]
            T_probe = len(probe["ids"])
            ablation_ids = probe["ids"]
            positions = list(range(second_off, T_probe - 1))

        baseline_nll, ablated_nll = ablation_nll_grid(
            model, cfg, ablation_ids, positions
        )
        delta_nll = ablated_nll - baseline_nll  # >0 ⇒ head was useful
        extras["pattern_score"] = scores
        extras["ablation_baseline_nll"] = np.full_like(scores, baseline_nll)
        extras["ablation_nll"] = ablated_nll.astype(np.float32)
        extras["ablation_delta_nll"] = delta_nll.astype(np.float32)
        # Combined score: each ranked separately to a percentile in [0, 1]
        # (1 = best). Take the *minimum* of the two percentiles — i.e., a head
        # is high-combined only when high in BOTH pattern AND ablation.
        def _pct_rank(M: np.ndarray) -> np.ndarray:
            flat = M.flatten()
            order = flat.argsort()
            r = np.empty_like(order, dtype=np.float32)
            r[order] = np.arange(len(order), dtype=np.float32)
            return (r / max(len(flat) - 1, 1)).reshape(M.shape)
        pat_pct = _pct_rank(scores)
        abl_pct = _pct_rank(delta_nll)
        combined_score = np.minimum(pat_pct, abl_pct)  # ∈ [0, 1]
        extras["pattern_pct"] = pat_pct
        extras["ablation_pct"] = abl_pct
        extras["combined_pct"] = combined_score

    # Write the CSV. Primary `score` column is the pattern score (preserved for
    # back-compat with multi_role_analysis that reads {slug}_score columns).
    _write_scores_csv(out_dir / "scores.csv", scores, extras)

    # top_heads.json: ranked by combined score when available, else by pattern.
    if combined_score is not None:
        top = _write_top_heads(
            combined_score, out_dir / "top_heads.json",
            pattern_score=scores, ablation_delta_nll=delta_nll.astype(np.float32),
        )
        # Also keep a pattern-only ranking on disk for inspection.
        _write_top_heads(scores, out_dir / "top_heads_pattern_only.json")
    else:
        top = _write_top_heads(scores, out_dir / "top_heads.json")
    return scores, top


def run_ablation_probe(slug: str, spec: dict, model, cfg, name: str, repo_id: str,
                        attn_cache: dict[tuple[str, str], np.ndarray]):
    """Ablation-based detector — successor or copy_suppression."""
    out_dir = RESULTS_ROOT / slug / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if spec["detector"] == "ablation_multi":
        probes = spec["probes"]
    else:
        probes = [spec["probe"]]
    thumb_probe = spec["probe"]
    T = len(thumb_probe["ids"])

    # Save the thumbnail attention (first / canonical probe).
    cache_key = (name, json.dumps(thumb_probe["ids"]))
    if cache_key in attn_cache:
        A = attn_cache[cache_key]
    else:
        A = capture_attention(model, thumb_probe["ids"], cfg.n_head)
        attn_cache[cache_key] = A
    np.save(out_dir / "attention.npy", A)
    _write_tokens_json(out_dir, name, repo_id, cfg, thumb_probe, T)

    L, H = cfg.n_layer, cfg.n_head

    if slug == "successor":
        # delta = baseline_logP(target) - ablated_logP(target). Avg over probes.
        per_probe_baseline = np.zeros((len(probes),), dtype=np.float32)
        per_probe_ablated = np.zeros((L, H, len(probes)), dtype=np.float32)
        for pi, probe in enumerate(probes):
            baseline_t, abl_t = ablation_grid(model, cfg, probe["ids"], [probe["target_id"]])
            per_probe_baseline[pi] = float(baseline_t[0])
            per_probe_ablated[:, :, pi] = abl_t[:, :, 0]
        delta = per_probe_baseline[None, None, :] - per_probe_ablated  # (L, H, P)
        delta_mean = delta.mean(axis=-1)
        # Score = mean Δ across probes (positive ⇒ head supports successor).
        extras = {f"baseline_logp_p{i}": np.full((L, H), per_probe_baseline[i], dtype=np.float32)
                  for i in range(len(probes))}
        for pi in range(len(probes)):
            extras[f"ablated_logp_p{pi}"] = per_probe_ablated[:, :, pi]
            extras[f"delta_p{pi}"] = delta[:, :, pi]
        _write_scores_csv(out_dir / "scores.csv", delta_mean, extras)
        top = _write_top_heads(delta_mean, out_dir / "top_heads.json")
        return delta_mean, top

    elif slug == "copy_suppression":
        probe = probes[0]
        target_id = probe["target_id"]
        distractor_id = probe["distractor_id"]
        baseline_lp, abl_lp = ablation_grid(
            model, cfg, probe["ids"], [target_id, distractor_id]
        )
        # delta_distractor = ablated - baseline. POSITIVE ⇒ head was suppressing distractor.
        delta_target = baseline_lp[0] - abl_lp[:, :, 0]      # (L, H)  — positive = head supported correct answer
        delta_distractor = abl_lp[:, :, 1] - baseline_lp[1]  # (L, H)  — positive = head was suppressing John
        score = delta_distractor                              # the headline detector
        extras = {
            "baseline_logp_target": np.full_like(score, baseline_lp[0]),
            "baseline_logp_distractor": np.full_like(score, baseline_lp[1]),
            "ablated_logp_target": abl_lp[:, :, 0],
            "ablated_logp_distractor": abl_lp[:, :, 1],
            "delta_target": delta_target,
            "delta_distractor": delta_distractor,
        }
        _write_scores_csv(out_dir / "scores.csv", score, extras)
        top = _write_top_heads(score, out_dir / "top_heads.json",
                               delta_target=delta_target, delta_distractor=delta_distractor)
        return score, top

    raise ValueError(f"unknown ablation slug: {slug}")


# %% ---------- Driver ----------
def run_one_model(name: str, repo_id: str, summary_rows: list[dict]):
    print(f"\n{'='*80}\n>>> {name}  ({repo_id})\n{'='*80}")
    model, cfg = load_model_from_repo(repo_id, device)
    print(f"L={cfg.n_layer}  H={cfg.n_head}  S={cfg.hyper_conn_n}  type={cfg.hyper_conn_type}")

    # In-process cache for attention captures so induction/duplicate share a forward pass.
    attn_cache: dict[tuple[str, str], np.ndarray] = {}

    for slug, spec in PROBES.items():
        print(f"\n-- {slug} ({spec['detector']}) --")
        _write_prompt_json(RESULTS_ROOT / slug, slug, spec)
        if spec["detector"] == "pattern":
            scores, top = run_pattern_probe(slug, spec, model, cfg, name, repo_id, attn_cache)
        else:
            scores, top = run_ablation_probe(slug, spec, model, cfg, name, repo_id, attn_cache)
        for entry in top:
            summary_rows.append({
                "head_type": slug, "model": name,
                "rank": entry["rank"], "layer": entry["layer"],
                "head": entry["head"], "score": entry["score"],
            })
        head_str = ", ".join(f"L{e['layer']}h{e['head']}({e['score']:.3f})" for e in top[:5])
        print(f"   top-5: {head_str}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def write_summary(summary_rows: list[dict]):
    out_path = RESULTS_ROOT / "summary.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["head_type", "model", "rank", "layer", "head", "score"])
        w.writeheader()
        for row in summary_rows:
            w.writerow({**row, "score": f"{row['score']:.6f}"})
    print(f"\nsaved {out_path}  ({len(summary_rows)} rows)")


if __name__ == "__main__":
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    for spec in MODELS:
        run_one_model(spec["name"], spec["repo_id"], summary_rows)
    write_summary(summary_rows)

# %%
