"""Base statistics of the TP=2 exchange on real activations (HF transformers, BF16 weights; several GPUs via
device_map=auto when the model does not fit one).  For every row-parallel projection (self_attn.o_proj, mlp.down_proj)
the two TP=2 partial sums are computed in FP32 (a = x[:, :K/2] W[:, :K/2]^T, b = the rest) and the BF16 output is
formed three ways:  TP1 bf16(a+b)  |  TP2 BF16 bf16(bf16(a)+bf16(b))  |  TP2 FP16 bf16(fp16(a)+fp16(b)).
Error = |output - (a+b)| / RMS(row of a+b).  Counts above thresholds are kept per chunk (for bootstrap CIs), per
projection, per layer and per language; the cancellation ratio r = (|a|+|b|)/|a+b| is summarised.
  python halfsums2.py MODEL texts.json out.json [n_chunks]"""
import json, re, sys, torch
from collections import defaultdict
from transformers import AutoModelForCausalLM
model_dir, tf, out = sys.argv[1:4]; nch = int(sys.argv[4]) if len(sys.argv) > 4 else 16
T = json.load(open(tf))["chunks"]; step = max(1, len(T) // nch); chunks = T[::step][:nch]
m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto").eval()
ARMS = ("tp1", "bf16", "fp16", "fp16s2"); THR = (1e-3, 1e-2, 3e-2)
per_chunk = []                     # per chunk: tokens, lang, counts[arm][thr]
by_proj = defaultdict(lambda: {a: {t: 0 for t in THR} for a in ARMS})
by_layer = defaultdict(lambda: {a: 0 for a in ARMS})   # counts > 1e-2
r_hist = defaultdict(int); n_el = [0]; cur = {}
gen = torch.Generator().manual_seed(0); samples = {a: [] for a in ARMS}; r_samp = []
def mk(name):
    layer = int(re.search(r"layers\.(\d+)\.", name).group(1)); proj = "o_proj" if name.endswith("o_proj") else "down_proj"
    def hook(mod, inp, outp):
        x = inp[0].reshape(-1, inp[0].shape[-1]).float(); W = mod.weight.float(); h = W.shape[1] // 2
        a = x[:, :h] @ W[:, :h].T; b = x[:, h:] @ W[:, h:].T; s = a + b
        rms = s.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
        outs = {"tp1": s.bfloat16().float(), "bf16": (a.bfloat16().float() + b.bfloat16().float()).bfloat16().float(),
                "fp16": (a.half().float() + b.half().float()).bfloat16().float(),
                "fp16s2": ((a * 0.25).half().float() + (b * 0.25).half().float()).bfloat16().float() * 4}
        r = ((a.abs() + b.abs()) / s.abs().clamp_min(1e-30)).flatten()
        n_el[0] += r.numel()
        for k in (8, 64, 512, 4096): r_hist[k] += int((r > k).sum())
        idx = torch.randint(0, r.numel(), (2048,), generator=gen).to(r.device)
        r_samp.extend(r[idx].clamp_max(1e9).tolist())
        for arm, o in outs.items():
            e = ((o - s).abs() / rms).flatten()
            samples[arm].extend(e[idx].tolist())
            for t in THR:
                c = int((e > t).sum()); cur[arm][t] += c; by_proj[proj][arm][t] += c
            by_layer[layer][arm] += int((e > 1e-2).sum())
    return hook
for name, mod in m.named_modules():
    if re.search(r"layers\.\d+\.(self_attn\.o_proj|mlp\.down_proj)$", name): mod.register_forward_hook(mk(name))
dev = next(m.parameters()).device
with torch.no_grad():
    for c in chunks:
        cur.clear(); cur.update({a: {t: 0 for t in THR} for a in ARMS})
        m(torch.tensor([c["ids"]], device=dev))
        per_chunk.append({"lang": c["lang"], "tokens": len(c["ids"]), "counts": {a: dict(v) for a, v in cur.items()}})
def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]
tok = sum(c["tokens"] for c in per_chunk)
def per_token(arm, t, cs): return sum(c["counts"][arm][t] for c in cs) / sum(c["tokens"] for c in cs)
import random; random.seed(0)
def boot(arm, t):
    v = sorted(per_token(arm, t, random.choices(per_chunk, k=len(per_chunk))) for _ in range(2000)); return [v[50], v[1949]]
def boot_ratio(t):
    v = []
    for _ in range(2000):
        cs = random.choices(per_chunk, k=len(per_chunk)); v.append(per_token("bf16", t, cs) / max(per_token("tp1", t, cs), 1e-9))
    v.sort(); return [v[50], v[1949]]
res = {"model": model_dir, "tokens": tok, "chunks": len(per_chunk), "exchanged_elements_per_token": n_el[0] // tok,
       "cancellation": {"frac_r_above": {k: v / n_el[0] for k, v in r_hist.items()}, "r_percentiles": {p: q(r_samp, p) for p in (0.5, 0.9, 0.99, 0.999, 0.9999)}},
       "err_percentiles": {a: {p: q(samples[a], p) for p in (0.5, 0.99, 0.999, 0.9999)} for a in ARMS},
       "per_token": {a: {t: {"mean": per_token(a, t, per_chunk), "ci95": boot(a, t)} for t in THR} for a in ARMS},
       "bf16_over_tp1_ratio": {t: {"mean": per_token("bf16", t, per_chunk) / per_token("tp1", t, per_chunk), "ci95": boot_ratio(t)} for t in THR},
       "fp16_over_tp1_ratio": {t: per_token("fp16", t, per_chunk) / per_token("tp1", t, per_chunk) for t in THR},
       "fp16s2_over_tp1_ratio": {t: per_token("fp16s2", t, per_chunk) / per_token("tp1", t, per_chunk) for t in THR},
       "by_projection_per_token": {p: {a: {t: v / tok for t, v in d.items()} for a, d in pa.items()} for p, pa in by_proj.items()},
       "by_language_per_token_1e-2": {l: {a: per_token(a, 1e-2, [c for c in per_chunk if c["lang"] == l]) for a in ARMS} for l in sorted({c["lang"] for c in per_chunk})},
       "by_layer_per_token_1e-2": {L: {a: v / tok for a, v in d.items()} for L, d in sorted(by_layer.items())}}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("tokens", "exchanged_elements_per_token", "per_token", "bf16_over_tp1_ratio", "fp16_over_tp1_ratio", "fp16s2_over_tp1_ratio")}, indent=1))
