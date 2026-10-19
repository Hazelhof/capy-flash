# The Qwen3.8-Flash-Next MTP layer in PyTorch on the NVIDIA checkpoint weights, dense part trainable. Validated
# against this engine (live acceptance within 0-2 points of this reference). Block structure after
# ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2 tools/mtp_refit/mtp_ref.py (Apache-2.0).
import json
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from safetensors import safe_open
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRMSNorm,
    Qwen4ExpTextRotaryEmbedding,
    apply_rotary_pos_emb,
)

M = os.environ.get("MTP_REFIT_CHECKPOINT", "")  # NVIDIA ModelOpt checkpoint dir (nvidia/Qwen3.8-Flash-Next-NVFP4)
bf = torch.bfloat16


class Snap:
    def __init__(self, p):
        self.p, self.map, self.f = p, json.load(open(f"{p}/model.safetensors.index.json"))["weight_map"], {}

    def get(self, n):
        fn = self.map[n]
        if fn not in self.f:
            self.f[fn] = safe_open(f"{self.p}/{fn}", "pt", device="cpu")
        return self.f[fn].get_tensor(n)


def text_config(experts_impl=None):
    c = json.load(open(f"{M}/config.json"))
    c = dict(c.get("text_config", c))
    c.update(num_hidden_layers=1, layer_types=["full_attention"], ple_layer_ids=[])
    cfg = Qwen4ExpTextConfig(**c)
    if experts_impl:
        cfg._experts_implementation = experts_impl
    return cfg


def fp8_block_dequant(w, s):
    o, i = w.shape
    s = s.float().repeat_interleave(128, 0)[:o].repeat_interleave(128, 1)[:, :i]
    return w.float() * s


def attention_sdpa(q, k, v):
    """Causal GQA attention through torch SDPA (O(n) memory): q [n,nh,hd] over k/v [n,nkv,hd], row r sees keys
    <= r. Same semantics as attention(q, k, v, causal) without diag (served draft steps attend depth-0 keys only)."""
    o = F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True, enable_gqa=True
    )
    return o[0].transpose(0, 1)


def attention(q, k, v, mask, diag=None):
    """q [n,nh,hd] over k/v [m,nkv,hd] with mask [n,m]; diag = (k,v) [n,nkv,hd] seen only by its own row."""
    n, nh, hd = q.shape
    m, nkv = k.shape[:2]
    with torch.autocast(q.device.type, enabled=False):
        qg = q.view(n, nkv, nh // nkv, hd).permute(1, 2, 0, 3).float()
        s = qg @ k.permute(1, 2, 0).float()[:, None] * hd**-0.5
        s = s.masked_fill(~mask, float("-inf"))
        if diag is not None:
            dk, dv = diag
            d = (qg * dk.permute(1, 0, 2).float()[:, None]).sum(-1, keepdim=True) * hd**-0.5
            s = torch.cat([s, d], -1)
        p = s.softmax(-1)
        o = p[..., :m] @ v.permute(1, 0, 2).float()[:, None]
        if diag is not None:
            o = o + p[..., m:] * dv.permute(1, 0, 2).float()[:, None]
        return o.permute(2, 0, 1, 3).reshape(n, nh, hd).to(q.dtype)


class Mtp(nn.Module):
    TRAIN_SKIP = (".experts.", "self_attn.indexer.")

    def __init__(self, cfg):
        super().__init__()
        H, S = cfg.hidden_size, cfg.hc_count
        self.cfg, self.H, self.S = cfg, H, S
        self.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(H, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(S * H, eps=cfg.rms_norm_eps)
        self.fc_embedding = nn.Linear(H, H, bias=False)
        self.fc_hidden = nn.Linear(H, H, bias=False)
        self.layers = nn.ModuleList([Qwen4ExpTextDecoderLayer(cfg, 0)])
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(cfg, use_combine=False)

    def trainable_names(self):
        return [n for n, _ in self.named_parameters() if not any(s in n for s in self.TRAIN_SKIP)]

    def block(self, e, h, pos, attend):
        n, S, H = e.shape[0], self.S, self.H
        en = self.pre_fc_norm_embedding(e)
        hn = self.pre_fc_norm_hidden(h).view(n, S, H)
        x = (self.fc_hidden(hn) + self.fc_embedding(en)[:, None]).flatten(-2)
        L = self.layers[0]
        a = L.self_attn
        m, hyper, inj = L.attn_hyper_connection(x)
        hd = self.cfg.head_dim
        q, gate = a.q_proj(m).view(n, -1, 2 * hd).chunk(2, dim=-1)
        q = a.q_norm(q)
        k = a.k_norm(a.k_proj(m).view(n, -1, hd))
        v = a.v_proj(m).view(n, -1, hd)
        cos, sin = self.rotary(m, pos.view(1, 1, -1).expand(3, 1, -1))
        q, k = apply_rotary_pos_emb(q, k, cos[0], sin[0], unsqueeze_dim=1)
        o = attend(q, k, v).reshape(n, -1)
        x = hyper + (a.o_proj(o * torch.sigmoid(gate.reshape(n, -1)))[:, None] * inj[..., None]).flatten(-2)
        m, hyper, inj = L.mlp_hyper_connection(x)
        x = hyper + (L.mlp(m[None])[0][:, None] * inj[..., None]).flatten(-2)
        return x, self.hyper_connection_mixer(x), k, v

    def unroll(self, embed, ids, h, pos0, depth, self_key=False, ckpt=False, sdpa=False):
        """Anchors = rows of h ([n,S*H], target wide residual at absolute positions pos0..pos0+n-1); ids [n+depth]
        = tokens at positions pos0..pos0+n-1+depth. Depth k: token ids[r+1+k], state h[r] (k=0) or the depth k-1
        multi-hidden, position pos0+r+k, attention over the depth-0 keys of rows <= r (teacher forcing: exact
        for the acceptance chain). Returns per-depth sample [n,H]."""
        n = h.shape[0]
        causal = None if sdpa else torch.ones(n, n, dtype=torch.bool, device=h.device).tril()
        out, state, ctx = [], h, None
        for k in range(depth):
            tok = ids[1 + k : 1 + k + n]
            pos = torch.arange(n, device=h.device) + pos0 + k
            if sdpa and not self_key:
                att = (
                    (lambda q, kk, vv: attention_sdpa(q, kk, vv))
                    if k == 0
                    else (lambda q, kk, vv, c=ctx: attention_sdpa(q, c[0], c[1]))
                )
            elif k == 0:
                att = lambda q, kk, vv: attention(q, kk, vv, causal)
            else:
                att = lambda q, kk, vv, c=ctx: attention(q, c[0], c[1], causal, (kk, vv) if self_key else None)
            if ckpt:
                state, sample, kk, vv = checkpoint(self.block, embed[tok], state, pos, att, use_reentrant=False)
            else:
                state, sample, kk, vv = self.block(embed[tok], state, pos, att)
            if k == 0:
                ctx = (kk, vv)
            out.append(sample)
        return out


def load(device, train=False, experts_impl=None, refit=None):
    """Mtp on `device` with the checkpoint weights (frozen BF16; trainable tensors fp32 when train), plus
    embed [V,H] and lm_head [V,H] BF16. refit = safetensors of trained mtp.* tensors to load on top."""
    cfg = text_config(experts_impl)
    snap = Snap(M)
    with torch.device("meta"):
        model = Mtp(cfg)
    model.to_empty(device=device)
    model.rotary = Qwen4ExpTextRotaryEmbedding(cfg).to(device)
    exp = model.layers[0].mlp.experts
    I = cfg.moe_intermediate_size
    over = {}
    if refit:
        with safe_open(refit, "pt", device="cpu") as f:
            over = {k: f.get_tensor(k) for k in f.keys()}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".experts." in n:
                continue
            src = over.get("mtp." + n)
            src = snap.get("mtp." + n) if src is None else src
            assert tuple(src.shape) == tuple(p.shape), (n, src.shape, p.shape)
            p.copy_(src)
        for e in range(cfg.num_experts):
            pre = f"mtp.layers.0.mlp.experts.{e}."
            w = {
                k: fp8_block_dequant(snap.get(pre + k + ".weight"), snap.get(pre + k + ".weight_scale_inv"))
                for k in ("gate_proj", "up_proj", "down_proj")
            }
            exp.gate_up_proj[e, :I].copy_(w["gate_proj"])
            exp.gate_up_proj[e, I:].copy_(w["up_proj"])
            exp.down_proj[e].copy_(w["down_proj"])
    model = model.to(bf)
    tn = set(model.trainable_names())
    for n, p in model.named_parameters():
        p.requires_grad_(train and n in tn)
        if train and n in tn:
            p.data = p.data.float()
    embed = snap.get("model.language_model.embed_tokens.weight").to(device, bf)
    head = snap.get("lm_head.weight").to(device, bf)
    return model, embed, head


def export(model, path, meta=None):
    from safetensors.torch import save_file

    params = dict(model.named_parameters())
    save_file(
        {"mtp." + n: params[n].detach().to(bf).cpu().contiguous() for n in model.trainable_names()},
        path,
        metadata={k: str(v) for k, v in (meta or {}).items()},
    )
