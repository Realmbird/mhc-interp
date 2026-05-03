"""Shared loaders for the Realmbird/mhc-781m-* checkpoints.

Each repo ships its own `model.py` + `hyper_conn/` package, so we have to
swap them in/out of `sys.modules` between models. Both `attention_patterns`
and `head_finder` use this.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def load_model_from_repo(repo_id: str, device: str) -> tuple[Any, Any]:
    """Pulls the repo, isolates its `model` / `hyper_conn` modules, and
    returns (model in eval mode on `device`, GPTConfig)."""
    local = snapshot_download(repo_id=repo_id)
    for k in list(sys.modules):
        if k == "model" or k == "hyper_conn" or k.startswith("hyper_conn."):
            del sys.modules[k]
    sys.path.insert(0, local)
    try:
        m = importlib.import_module("model")
    finally:
        sys.path.pop(0)
    cfg = m.GPTConfig(**json.load(open(f"{local}/config.json")))
    g = m.GPT(cfg)
    g.load_state_dict(load_file(f"{local}/model.safetensors"))
    g.eval().to(device)
    return g, cfg


def get_token_corpus(n_tokens: int, seq_len: int = 512, source: str = "dolma", cache_path=None):
    """Return a (B, seq_len) gpt2-tokenized tensor with B*seq_len ≥ n_tokens.

    `source` ∈ {"dolma", "wikitext"}. Dolma is in-distribution for the
    Realmbird/mhc-781m-* checkpoints (they trained on a subset of Dolma v1_7);
    wikitext is the fast/clean fallback.

    Cached on disk per (source, n_tokens, seq_len).
    """
    from pathlib import Path
    cache_path = Path(cache_path) if cache_path else Path.home() / ".cache" / "mhc_interp_tokens"
    cache_path.mkdir(parents=True, exist_ok=True)
    f = cache_path / f"{source}_n{n_tokens}_s{seq_len}.pt"
    if f.exists():
        return torch.load(f)

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")

    ids: list[int] = []
    if source == "dolma":
        # The official allenai/dolma loader is a script-based loader that
        # modern `datasets` rejects. Use monology/pile-uncopyrighted instead —
        # The Pile is a major component of Dolma's training mix (CC + Wiki +
        # books + code), so it's effectively in-distribution for these models.
        ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        for record in ds:
            chunk = tok.encode(record["text"])
            ids.extend(chunk)
            if len(ids) >= n_tokens:
                break
    elif source == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
        ids = tok.encode(text)
    else:
        raise ValueError(f"unknown corpus source: {source}")

    if len(ids) < n_tokens:
        raise ValueError(f"corpus {source} only yielded {len(ids)} tokens, requested {n_tokens}")
    ids = ids[:n_tokens]
    n_seqs = n_tokens // seq_len
    arr = torch.tensor(ids[: n_seqs * seq_len], dtype=torch.long).view(n_seqs, seq_len)
    torch.save(arr, f)
    return arr


@torch.no_grad()
def attn_from_qkv(qkv: torch.Tensor, n_head: int) -> torch.Tensor:
    """qkv: (B, T, 3*D) from c_attn → causal-masked attention weights (B, H, T, T)."""
    B, T, D3 = qkv.shape
    D = D3 // 3
    hs = D // n_head
    q, k, _ = qkv.split(D, dim=2)
    q = q.view(B, T, n_head, hs).transpose(1, 2)
    k = k.view(B, T, n_head, hs).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
    causal = torch.triu(torch.ones(T, T, device=att.device, dtype=torch.bool), diagonal=1)
    att = att.masked_fill(causal, float("-inf"))
    return F.softmax(att, dim=-1)
