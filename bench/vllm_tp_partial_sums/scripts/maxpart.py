"""Largest |partial sum| of the TP=2 halves of every row-parallel projection (FP16 range check, max 65504).
python maxpart.py MODEL texts.json [n_chunks]"""

import json
import re
import sys
import torch
from transformers import AutoModelForCausalLM

model_dir, tf = sys.argv[1:3]
nch = int(sys.argv[3]) if len(sys.argv) > 3 else 8
T = json.load(open(tf))["chunks"]
chunks = T[:: max(1, len(T) // nch)][:nch]
m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto").eval()
mx = {"o_proj": 0.0, "down_proj": 0.0}
where = {}


def mk(name):
    proj = "o_proj" if name.endswith("o_proj") else "down_proj"

    def hook(mod, inp, outp):
        x = inp[0].reshape(-1, inp[0].shape[-1]).float()
        W = mod.weight.float()
        h = W.shape[1] // 2
        v = max(float((x[:, :h] @ W[:, :h].T).abs().max()), float((x[:, h:] @ W[:, h:].T).abs().max()))
        if v > mx[proj]:
            mx[proj] = v
            where[proj] = name

    return hook


for name, mod in m.named_modules():
    if re.search(r"layers\.\d+\.(self_attn\.o_proj|mlp\.down_proj)$", name):
        mod.register_forward_hook(mk(name))
with torch.no_grad():
    for c in chunks:
        m(torch.tensor([c["ids"]], device=next(m.parameters()).device))
print(json.dumps({"model": model_dir, "max_abs_partial": mx, "at": where}))
