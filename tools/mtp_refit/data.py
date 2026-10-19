# Sequences of the compact dump (score-windows) + the generation jsonl. A sequence = tokens [N], hidden bf16
# [N-1, S*H] (positions 0..N-2), top-k [N-1] (row p = the model's distribution of token p+1), prompt_tokens P.
import json
import os
import numpy as np
import torch
from compact import read_hidden, read_topk


def sequences(dump_dir, gen_jsonl, meta_jsonl=None):
    gens = [json.loads(l) for l in open(gen_jsonl)]
    meta = [json.loads(l) for l in open(meta_jsonl)] if meta_jsonl else [{}] * len(gens)
    for i, (g, m) in enumerate(zip(gens, meta)):
        d = f"{dump_dir}/seq{i:03d}" if os.path.isdir(f"{dump_dir}/seq{i:03d}") else f"{dump_dir}/seq{i:05d}"
        if not os.path.exists(f"{d}/topk.bin"):
            continue
        yield i, d, g["prompt_tokens"], m


def load_seq(d, device):
    toks = torch.from_numpy(np.load(f"{d}/tokens.npy")).to(device)
    pos, h = read_hidden(f"{d}/hidden_L47.bin")
    assert (pos == np.arange(len(pos))).all()
    tk = read_topk(f"{d}/topk.bin")
    assert (tk["pos"] == np.arange(len(tk))).all()
    h = torch.from_numpy(h.copy()).to(device).view(torch.bfloat16)
    return toks, h, torch.from_numpy(tk["ids"].copy()).to(device), torch.from_numpy(tk["lp"].copy()).to(device)
