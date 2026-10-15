# Copied unchanged (below this header) from sybrix/qwen3.8-flash-next-ninfer @a2fd69bb, flash-next-reference/gen_synthetic.py
# Copyright sybrix, Apache License 2.0 (see the repository LICENSE / NOTICE).
"""Write a small random Qwen3.8-Flash-Next checkpoint in the NVIDIA ModelOpt NVFP4 layout.

Real per-layer geometry (hidden 2560, 512 NVFP4 experts, GDN 16/48 heads, 24/2 attention heads,
4 hyper-connection streams, PLE at block 1) with only `--layers` Text blocks and a tiny n-gram
table, so the converter, NInfer and the layer-streaming reference can be compared end to end
without the 133 GB checkpoint. Tokenizer and template files are copied from a real snapshot.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15


def splitmix64(v):
    v = (v + GAMMA) & MASK64
    v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & MASK64
    return (v ^ (v >> 31)) & MASK64


def is_prime(v):
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def ple_constants(t, ple_index):
    vocab, order, per = t["vocab_size"], t["ngram_size"], t["heads_per_ngram"]
    bound = max(1, ((1 << 63) - 1) // vocab // 2)
    seed = t["seed"] + 10007 * ple_index
    mult = [2 * (splitmix64((seed + GAMMA * (i + 1)) & MASK64) % bound) + 1 for i in range(order)]
    heads = (order - 1) * per
    prime, sizes, offsets, total = t["ngram_vocab_size_base"] - 1, [], [], 0
    for _ in range(ple_index * heads):
        prime += 1
        while not is_prime(prime):
            prime += 1
    for _ in range(heads):
        prime += 1
        while not is_prime(prime):
            prime += 1
        sizes.append(prime)
        offsets.append(total)
        total += prime
    div = t["make_ngram_vocab_size_divisible_by"]
    return mult, sizes, offsets, -(-total // div) * div


def bf16(shape, std, gen):
    return (torch.randn(shape, generator=gen) * std).to(torch.bfloat16)


def nvfp4(rows, cols, gen, rng):
    packed = torch.from_numpy(rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8))
    # E4M3 block scales near 1.0 (exponent 7): words 0x34..0x3C.
    scale = torch.from_numpy(rng.integers(0x34, 0x3C, size=(rows, cols // 16), dtype=np.uint8))
    return {
        "weight": packed,
        "weight_scale": scale.view(torch.float8_e4m3fn),
        "weight_scale_2": torch.tensor(0.02 / 3.0 * float(rng.uniform(0.5, 1.5)), dtype=torch.float32),
        "input_scale": torch.tensor(float(rng.uniform(0.01, 0.1)), dtype=torch.float32),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True, help="real snapshot for config and tokenizer files")
    p.add_argument("--out", required=True)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ngram-base", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--lm-head-std", type=float, default=0.02)
    p.add_argument("--mtp", action="store_true", help="add the MTP head (FP8_PB_WO experts)")
    p.add_argument("--vision", action="store_true", help="add the (Qwen3.5-shaped) vision tower")
    args = p.parse_args()
    src, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "chat_template.jinja", "generation_config.json", "preprocessor_config.json",
                 "video_preprocessor_config.json"):
        if (src / name).exists():
            shutil.copy(src / name, out / name)
    config = json.loads((src / "config.json").read_text())
    config.pop("quantization_config", None)
    t = config["text_config"]
    n = args.layers
    t["num_hidden_layers"] = n
    t["layer_types"] = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n)]
    t["ngram_vocab_size_base"] = args.ngram_base
    t["mtp_num_hidden_layers"] = 1 if args.mtp else 0
    (out / "config.json").write_text(json.dumps(config, indent=2))

    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    H, S, R = t["hidden_size"], t["hc_count"], t["hc_lowrank"]
    V, E, I = t["vocab_size"], t["num_experts"], t["moe_intermediate_size"]
    Is = t["shared_expert_intermediate_size"]
    pre = "model.language_model."
    tensors = {}
    tensors[pre + "embed_tokens.weight"] = bf16((V, H), 0.02, gen)
    tensors["lm_head.weight"] = bf16((V, H), args.lm_head_std, gen)

    def gated(prefix, inject=True):
        tensors[prefix + "hc_norm.weight"] = bf16((S * H,), 0.05, gen)
        tensors[prefix + "input_mix_weight_down.weight"] = bf16((R, S * H), 0.02, gen)
        tensors[prefix + "input_mix_weight_up.weight"] = bf16((S * H, R), 0.02, gen)
        if inject:
            tensors[prefix + "block_inject_weight.weight"] = bf16((S, S * H), 0.02, gen)

    gated(pre + "hyper_connection_mixer.", inject=False)
    nk, kd = t["linear_num_key_heads"], t["linear_key_head_dim"]
    nv, vd = t["linear_num_value_heads"], t["linear_value_head_dim"]
    qh, kvh, hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    ih, ihd = t["indexer_n_heads"], t["indexer_head_dim"]
    ple_blocks = [b - 1 for b in t["ple_layer_ids"]]
    def attention(a):
        tensors[a + "q_proj.weight"] = bf16((2 * qh * hd, H), 0.02, gen)
        tensors[a + "k_proj.weight"] = bf16((kvh * hd, H), 0.02, gen)
        tensors[a + "v_proj.weight"] = bf16((kvh * hd, H), 0.02, gen)
        tensors[a + "o_proj.weight"] = bf16((H, qh * hd), 0.02, gen)
        tensors[a + "q_norm.weight"] = bf16((hd,), 0.05, gen)
        tensors[a + "k_norm.weight"] = bf16((hd,), 0.05, gen)
        tensors[a + "indexer.index_qk_proj.weight"] = bf16(((ih + 1) * ihd, H), 0.02, gen)
        tensors[a + "indexer.q_layernorm.weight"] = bf16((ihd,), 0.05, gen)
        tensors[a + "indexer.k_layernorm.weight"] = bf16((ihd,), 0.05, gen)

    def mlp(m, fp8=False):
        tensors[m + "gate.weight"] = bf16((E, H), 0.05, gen)
        tensors[m + "shared_expert.gate_proj.weight"] = bf16((Is, H), 0.02, gen)
        tensors[m + "shared_expert.up_proj.weight"] = bf16((Is, H), 0.02, gen)
        tensors[m + "shared_expert.down_proj.weight"] = bf16((H, Is), 0.02, gen)
        tensors[m + "shared_expert_gate.weight"] = bf16((1, H), 0.02, gen)
        for e in range(E):
            for proj, (rows, cols) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
                if fp8:
                    # FP8_PB_WO: E4M3 weight, one BF16 multiplier per 128x128 tile.
                    scale = torch.empty(-(-rows // 128), -(-cols // 128)).uniform_(
                        2e-4, 6e-4, generator=gen)
                    w = torch.randn(rows, cols, generator=gen).clamp(-4, 4) * 30
                    tensors[f"{m}experts.{e}.{proj}.weight"] = w.to(torch.float8_e4m3fn)
                    tensors[f"{m}experts.{e}.{proj}.weight_scale_inv"] = scale.to(torch.bfloat16)
                    continue
                for key, value in nvfp4(rows, cols, gen, rng).items():
                    tensors[f"{m}experts.{e}.{proj}.{key}"] = value

    shards = []
    for i in range(n):
        lp = f"{pre}layers.{i}."
        gated(lp + "attn_hyper_connection.")
        gated(lp + "mlp_hyper_connection.")
        if t["layer_types"][i] == "linear_attention":
            g = lp + "linear_attn."
            conv = 2 * nk * kd + nv * vd
            tensors[g + "A_log"] = torch.empty(nv).uniform_(1, 16, generator=gen).log().to(torch.bfloat16)
            tensors[g + "dt_bias"] = torch.ones(nv, dtype=torch.bfloat16)
            tensors[g + "conv1d.weight"] = bf16((conv, 1, 4), 0.2, gen)
            tensors[g + "in_proj_a.weight"] = bf16((nv, H), 0.02, gen)
            tensors[g + "in_proj_b.weight"] = bf16((nv, H), 0.02, gen)
            tensors[g + "in_proj_qkv.weight"] = bf16((conv, H), 0.02, gen)
            tensors[g + "in_proj_z.weight"] = bf16((nv * vd, H), 0.02, gen)
            tensors[g + "norm.weight"] = (1.0 + torch.randn(vd, generator=gen) * 0.05).to(torch.bfloat16)
            tensors[g + "out_proj.weight"] = bf16((H, nv * vd), 0.02, gen)
        else:
            attention(lp + "self_attn.")
        mlp(lp + "mlp.")
        if i in ple_blocks:
            index = ple_blocks.index(i)
            mult, sizes, offsets, rows = ple_constants(t, index)
            q = lp + "ple."
            width = t["ple_embed_dim"] // ((t["ngram_size"] - 1) * t["heads_per_ngram"])
            tensors[q + "conv1d.weight"] = bf16((S * H, 1, t["ple_conv_kernel_size"]), 0.2, gen)
            tensors[q + "key_proj.weight"] = bf16((S * H, t["ple_embed_dim"]), 0.02, gen)
            tensors[q + "value_proj.weight"] = bf16((H, t["ple_embed_dim"]), 0.02, gen)
            for norm in ("norm_conv", "norm_key", "norm_query"):
                tensors[q + norm + ".weight"] = bf16((S * H,), 0.05, gen)
            emb = q + "ple_embedding."
            tensors[emb + "layer_multipliers"] = torch.tensor(mult, dtype=torch.int64)
            tensors[emb + "ngram_heads_vocab_sizes"] = torch.tensor(sizes, dtype=torch.int64)
            tensors[emb + "ngram_heads_offsets"] = torch.tensor(offsets, dtype=torch.int64)
            tensors[emb + "ngram_embedding.weight_scale"] = torch.tensor([0.015625], dtype=torch.bfloat16)
            parts = 128
            assert rows % parts == 0, rows
            table = torch.randn(rows, width, generator=gen).clamp(-4, 4).to(torch.float8_e4m3fn)
            for s in range(parts):
                chunk = rows // parts
                tensors[f"{emb}ngram_embedding.shard_{s}.weight"] = table[s * chunk:(s + 1) * chunk].clone()
        # Spill a file per layer to bound host memory.
        if i != n - 1:
            shards.append(dict(tensors))
            tensors = {}
    if args.vision:
        shards.append(tensors)
        tensors = {}
        vc = config["vision_config"]
        vh, vi, depth = vc["hidden_size"], vc["intermediate_size"], vc["depth"]
        ps, pt, merge = vc["patch_size"], vc["temporal_patch_size"], vc["spatial_merge_size"]
        out_h, mh = vc["out_hidden_size"], vh * merge * merge
        v = "model.visual."
        tensors[v + "patch_embed.proj.weight"] = bf16((vh, 3, pt, ps, ps), 0.02, gen)
        tensors[v + "patch_embed.proj.bias"] = bf16((vh,), 0.02, gen)
        tensors[v + "pos_embed.weight"] = bf16((vc["num_position_embeddings"], vh), 0.02, gen)
        for i in range(depth):
            b = f"{v}blocks.{i}."
            for norm in ("norm1", "norm2"):
                tensors[b + norm + ".weight"] = (1.0 + torch.randn(vh, generator=gen) * 0.05).to(torch.bfloat16)
                tensors[b + norm + ".bias"] = bf16((vh,), 0.02, gen)
            tensors[b + "attn.qkv.weight"] = bf16((3 * vh, vh), 0.02, gen)
            tensors[b + "attn.qkv.bias"] = bf16((3 * vh,), 0.02, gen)
            tensors[b + "attn.proj.weight"] = bf16((vh, vh), 0.02, gen)
            tensors[b + "attn.proj.bias"] = bf16((vh,), 0.02, gen)
            tensors[b + "mlp.linear_fc1.weight"] = bf16((vi, vh), 0.02, gen)
            tensors[b + "mlp.linear_fc1.bias"] = bf16((vi,), 0.02, gen)
            tensors[b + "mlp.linear_fc2.weight"] = bf16((vh, vi), 0.02, gen)
            tensors[b + "mlp.linear_fc2.bias"] = bf16((vh,), 0.02, gen)
        tensors[v + "merger.norm.weight"] = (1.0 + torch.randn(vh, generator=gen) * 0.05).to(torch.bfloat16)
        tensors[v + "merger.norm.bias"] = bf16((vh,), 0.02, gen)
        tensors[v + "merger.linear_fc1.weight"] = bf16((mh, mh), 0.02, gen)
        tensors[v + "merger.linear_fc1.bias"] = bf16((mh,), 0.02, gen)
        tensors[v + "merger.linear_fc2.weight"] = bf16((out_h, mh), 0.02, gen)
        tensors[v + "merger.linear_fc2.bias"] = bf16((out_h,), 0.02, gen)
    if args.mtp:
        shards.append(tensors)
        tensors = {}
        tensors["mtp.pre_fc_norm_embedding.weight"] = bf16((H,), 0.05, gen)
        tensors["mtp.pre_fc_norm_hidden.weight"] = bf16((S * H,), 0.05, gen)
        tensors["mtp.fc_embedding.weight"] = bf16((H, H), 0.02, gen)
        tensors["mtp.fc_hidden.weight"] = bf16((H, H), 0.02, gen)
        gated("mtp.hyper_connection_mixer.", inject=False)
        gated("mtp.layers.0.attn_hyper_connection.")
        gated("mtp.layers.0.mlp_hyper_connection.")
        attention("mtp.layers.0.self_attn.")
        mlp("mtp.layers.0.mlp.", fp8=True)
    shards.append(tensors)
    weight_map = {}
    for k, shard in enumerate(shards):
        name = f"model-{k + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file({key: v.contiguous() for key, v in shard.items()}, str(out / name))
        weight_map.update({key: name for key in shard})
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    print(f"wrote {len(weight_map)} tensors in {len(shards)} files to {out}")


if __name__ == "__main__":
    main()
