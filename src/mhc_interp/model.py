# %%
import sys
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# mhc-lite isn't pip-installed; expose its `model` and `hyper_conn` modules.
sys.path.insert(0, "/home/chriskino/mhc-lite")
from model import GPT, GPTConfig

REPO_ID = "wgpeng/mhc-780m"
CKPT_FILE = "best_ckpt_large.pt"

# Repo only ships best_ckpt_large.pt — no config.json. The nanoGPT-style
# checkpoint stores GPTConfig args under ckpt["model_args"].
ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=CKPT_FILE)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

cfg = GPTConfig(**ckpt["model_args"])
model = GPT(cfg)

state_dict = ckpt["model"]
prefix = "_orig_mod."
for k in list(state_dict.keys()):
    if k.startswith(prefix):
        state_dict[k[len(prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Loaded {REPO_ID}: {n_params:.1f}M params")
print(f"  iter_num      = {ckpt.get('iter_num')}")
print(f"  best_val_loss = {ckpt.get('best_val_loss')}")
print(f"  model_args    = {ckpt['model_args']}")

# %%
tokenizer = AutoTokenizer.from_pretrained("gpt2")
prompt = "Language modeling is "
ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=100, temperature=1.0, top_k=None)
print(tokenizer.decode(out[0].tolist(), skip_special_tokens=True))
# %%
model