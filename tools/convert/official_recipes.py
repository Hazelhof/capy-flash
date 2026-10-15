"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

from .methods import (
    cast_direct,
    fp8_row_maxabs,
    grouped_absmax,
    import_encoded,
    nvfp4_mse,
    nvfp4_mse_stacked,
)

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"


def _assign(recipe, name, format, *, source=None):
    method = grouped_absmax if format in (Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(name, format=format, method=method, source=source)


def _optional(model, recipe):
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if name.startswith("vision/"):
            if name == "vision/patch_embedding":
                format = Q6
            elif name.startswith("vision/merger/"):
                format = Q8
            elif name.endswith(
                ("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")
            ):
                format = Q4
            else:
                format = Q5
            _assign(recipe, name, format)
        elif name.startswith(("mtp/", "dflash/", "dflash2/")):
            if name.endswith(
                (
                    "/moe/router",
                    "/moe/shared_score",
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            _assign(recipe, name, Q8)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _dflash2_nvfp4_gate_up(model, recipe):
    """Store each DFlash2 layer's MLP gate/up parent as NVFP4 with 16-bit activations."""
    if "dflash2" not in model.components:
        return
    layers = model.components["dflash2"]["config"]["num_hidden_layers"]
    for layer in range(layers):
        recipe.assign(
            [f"dflash2/layers/{layer}/mlp/gate", f"dflash2/layers/{layer}/mlp/up"],
            format="nvfp4",
            method=nvfp4_mse,
            activation_policy="A16Only",
        )


def _mtp_nvfp4_layer(model, recipe):
    """Store the MTP layer's MLP gate/up, MLP down and attention output as NVFP4 with 16-bit activations.

    Quantized from the source's MTP weights by nvfp4_mse. At tp 2 their halves ([17408,5120],
    [5120,8704], [5120,3072]) run the NVFP4 linear routes the target's MLP and attention output use.
    The packed attention input projection (Q|K|Gate|V) and mtp/input_projection stay Q8: the MTP
    layer runs its attention input as a plain linear, and NVFP4 has no [7168,5120] linear half."""
    if "mtp" not in model.components:
        return
    for names in (
        ["mtp/layers/0/mlp/gate", "mtp/layers/0/mlp/up"],
        "mtp/layers/0/mlp/down",
        "mtp/layers/0/attention/output",
    ):
        recipe.assign(names, format="nvfp4", method=nvfp4_mse, activation_policy="A16Only")


def _dense_groupwise(model, recipe, vocabulary):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", vocabulary)
    _assign(recipe, "text/output_head", vocabulary)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        if name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
                "/mlp/gate",
                "/mlp/up",
            )
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format)


def qwen3_6_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)


def qwen3_8_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8)


def qwen3_6_35b_a3b(model, recipe, sources):
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            layer = int(name.split("/")[2])
            format = (
                (Q6 if layer in (34, 38, 39) else Q5) if name.endswith("/down") else Q4
            )
        else:
            format = Q8
        _assign(recipe, name, format)


def qwen3_6_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        layer = int(name.split("/")[2])
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        direct = (
            ("/attention/" in name and not name.endswith("/output") and layer < 24)
            or (name.endswith("/attention/output") and layer in (3, 7))
            or (name.endswith("/gdn/output") and layer == 4)
        )
        if direct:
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=model.source(name, quantized, "nvfp4"),
            activation_policy="AllowA4",
        )


def qwen3_8_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name, source=source)
            continue
        layer = int(name.split("/")[2]) if name.startswith("text/layers/") else -1
        format = "nvfp4" if "/mlp/" in name and layer < 56 else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


# Dense projections the NVFP4-dense Flash-Next recipe stores as NVFP4: the attention and GDN
# input/output projections, the shared expert and the output head. Their tensor-parallel halves are
# the per-rank GEMV shapes measured for F0-B ([6656,2560], [8192,2560], [2560,3072], [640,2560],
# [124160,2560]). Hyper-connection mixers, the QSA indexer, the PLE projections and the MTP stem
# projections stay Q8: their format is not decided, and the hyper-connection down projection
# ([320|4, 10240]) cannot be NVFP4 at all (N % 128 != 0).
_FLASH_NEXT_NVFP4_GROUPS = {
    "attention": (
        "/attention/query",
        "/attention/key",
        "/attention/gate",
        "/attention/value",
        "/attention/output",
    ),
    "gdn": ("/gdn/query", "/gdn/key", "/gdn/value", "/gdn/z", "/gdn/output"),
    "shared": ("/moe/shared/gate", "/moe/shared/up", "/moe/shared/down"),
}
_FLASH_NEXT_NVFP4_DENSE = tuple(s for group in _FLASH_NEXT_NVFP4_GROUPS.values() for s in group)

# T103: dense projections a Flash-Next recipe may keep in BF16 (as the checkpoint, GB300 and Strata):
# the QSA indexer q|k, every hyper-connection mixer (block attn_hc/ffn_hc and the text/MTP hc_head,
# Down|inject and Up) and the GDN q|k|v|z|out projections. The engine runs them through the generic
# BF16 linear (ops/linear/bf16/bf16_generic.cu) and the hyper-connection general route.
_FLASH_NEXT_BF16_GROUPS = {
    "indexer": ("/attention/indexer_query", "/attention/indexer_key"),
    "hc": tuple(
        f"{mixer}/{role}"
        for mixer in ("attn_hc", "ffn_hc", "hc_head")
        for role in ("down", "inject", "up")
    ),
    "gdn": _FLASH_NEXT_NVFP4_GROUPS["gdn"],
}


def _flash_next(
    model, recipe, *, nvfp4_dense: bool, nvfp4_groups=None, nvfp4_head=None, bf16_groups=()
):
    """Qwen3.8-Flash-Next (Qwen4Exp) from the ModelOpt NVFP4 checkpoint.

    Routed Text experts keep their ModelOpt NVFP4 codes, block scales, weight divisors and
    activation divisors (``import_encoded``, AllowA4). Each layer's routed gate/up and down
    projections become one parent each, a stack with one divisor per source matrix, so an expert
    is a fixed row range of its bank: in the code plane, in the swizzled scale plane (each source
    covers whole 128-row tiles) and one FP32 divisor word per projection. The MTP layer's block-FP8
    experts are re-encoded to NVFP4 by nvfp4_mse_stacked, so they have the same per-matrix divisor
    layout (A16Only: the checkpoint has no activation calibration for them; ninfer-ext instead
    stores one absmax divisor for the whole MTP bank). The n-gram table keeps its FP8 codes under the shared
    multiplier. Routers, shared-expert gates, norms and small vectors stay direct; the token
    embedding is Q8.

    Dense projections are Q8 with a Q6 output head, exactly as ninfer-ext's recipe of the same
    name, unless ``nvfp4_dense``: then the projections in _FLASH_NEXT_NVFP4_DENSE and the output
    head are NVFP4 (nvfp4_mse, A16Only) and the GDN a/b projections stay Q8 in their own parent.
    ``nvfp4_groups`` (names of _FLASH_NEXT_NVFP4_GROUPS) and ``nvfp4_head`` select a subset instead
    (mixed recipes); the selection applies to the Text and MTP layers alike. ``bf16_groups``
    (names of _FLASH_NEXT_BF16_GROUPS, T103) keep those projections in BF16 (cast_direct), Text
    and MTP alike; they take precedence over ``nvfp4_groups``.
    """
    if nvfp4_groups is None:
        nvfp4_groups = tuple(_FLASH_NEXT_NVFP4_GROUPS) if nvfp4_dense else ()
    if nvfp4_head is None:
        nvfp4_head = nvfp4_dense
    nvfp4_suffixes = tuple(s for group in nvfp4_groups for s in _FLASH_NEXT_NVFP4_GROUPS[group])
    bf16_suffixes = tuple(s for group in bf16_groups for s in _FLASH_NEXT_BF16_GROUPS[group])
    if model.config.get("model_type") != "qwen4_exp_text":
        raise ValueError("this official recipe requires Qwen4Exp mathematics")
    # Vision's mixed groupwise assignment; the loop below restates every Text and MTP choice.
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    if nvfp4_head:
        recipe.assign(
            "text/output_head", format="nvfp4", method=nvfp4_mse, activation_policy="A16Only"
        )
    else:
        _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if name.startswith("vision/") or not (parameter.projection or name.endswith("/ple/table")):
            continue
        if name.endswith("/ple/table"):
            recipe.assign(name, format=FP8, method=import_encoded, source=parameter.source)
            continue
        if name.endswith(("/moe/router", "/moe/shared_score")) or name in (
            "text/token_embedding",
            "text/output_head",
        ):
            continue
        if "/moe/experts/" in name:
            if name.startswith("text/"):
                recipe.assign(
                    name,
                    format="nvfp4",
                    method=import_encoded,
                    source=parameter.source,
                    activation_policy="AllowA4",
                )
            else:
                recipe.assign(
                    name, format="nvfp4", method=nvfp4_mse_stacked, activation_policy="A16Only"
                )
            continue
        if bf16_suffixes and name.endswith(bf16_suffixes):
            _assign(recipe, name, "bf16")
            continue
        if nvfp4_suffixes and name.endswith(nvfp4_suffixes):
            recipe.assign(name, format="nvfp4", method=nvfp4_mse, activation_policy="A16Only")
            continue
        _assign(recipe, name, Q8)
    # Each layer's gate/up and down banks become one parent so an expert is a fixed row range of
    # its bank, addressed by the expert cache as a few contiguous spans.
    for names in model.packing_groups:
        if all("/moe/experts/" in name for name in names):
            recipe.group(names)


def qwen3_8_flash_next_nvfp4(model, recipe, sources):
    """Flash-Next, NVFP4 experts, Q8 dense projections, Q6 head (ninfer-ext's choices)."""
    _flash_next(model, recipe, nvfp4_dense=False)


def qwen3_8_flash_next_nvfp4_dense(model, recipe, sources):
    """Flash-Next, NVFP4 experts, NVFP4 attention/GDN/shared-expert projections and head."""
    _flash_next(model, recipe, nvfp4_dense=True)


def qwen3_8_flash_next_nvfp4_mix(model, recipe, sources):
    """Flash-Next, NVFP4 experts, NVFP4 attention/GDN projections and head, Q8 shared expert (T26b)."""
    _flash_next(model, recipe, nvfp4_dense=False, nvfp4_groups=("attention", "gdn"), nvfp4_head=True)


def qwen3_8_flash_next_nvfp4_gdn(model, recipe, sources):
    """Flash-Next, NVFP4 experts, NVFP4 GDN projections (q/k/v/z/out), everything else as the Q8 recipe (T26c)."""
    _flash_next(model, recipe, nvfp4_dense=False, nvfp4_groups=("gdn",), nvfp4_head=False)


def _flash_next_g_bf16(*groups, gdn_q8=False):
    """G (NVFP4 GDN, Q8 dense, Q6 head) with the T103 BF16 groups; BF16 GDN replaces NVFP4 GDN,
    and ``gdn_q8`` keeps the GDN projections Q8 (as the rest of the dense) instead of NVFP4."""

    def recipe_function(model, recipe, sources):
        _flash_next(
            model,
            recipe,
            nvfp4_dense=False,
            nvfp4_groups=() if ("gdn" in groups or gdn_q8) else ("gdn",),
            nvfp4_head=False,
            bf16_groups=groups,
        )

    gdn = " and Q8 GDN" if gdn_q8 else ""
    recipe_function.__doc__ = f"Flash-Next G with {', '.join(groups)} projections in BF16{gdn} (T103)."
    return recipe_function


qwen3_8_flash_next_nvfp4_gdn_idxbf16 = _flash_next_g_bf16("indexer")
qwen3_8_flash_next_nvfp4_gdn_hcbf16 = _flash_next_g_bf16("hc")
qwen3_8_flash_next_nvfp4_gdnbf16 = _flash_next_g_bf16("gdn")
qwen3_8_flash_next_nvfp4_gdn_idxhcbf16 = _flash_next_g_bf16("indexer", "hc")
qwen3_8_flash_next_nvfp4_allbf16 = _flash_next_g_bf16("indexer", "hc", "gdn")
# Valerio's T103 candidate: indexer and hyper-connections BF16, GDN Q8, everything else as G.
qwen3_8_flash_next_nvfp4_idxhcbf16_gdnq8 = _flash_next_g_bf16("indexer", "hc", gdn_q8=True)
# T104 (Valerio): indexer BF16 + GDN Q8, hyper-connections Q8 as G (cheapest candidate).
qwen3_8_flash_next_nvfp4_idxbf16_gdnq8 = _flash_next_g_bf16("indexer", gdn_q8=True)


RECIPES = {
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
    "qwen3_8_flash_next_nvfp4": qwen3_8_flash_next_nvfp4,
    "qwen3_8_flash_next_nvfp4_dense": qwen3_8_flash_next_nvfp4_dense,
    "qwen3_8_flash_next_nvfp4_mix": qwen3_8_flash_next_nvfp4_mix,
    "qwen3_8_flash_next_nvfp4_gdn": qwen3_8_flash_next_nvfp4_gdn,
    "qwen3_8_flash_next_nvfp4_gdn_idxbf16": qwen3_8_flash_next_nvfp4_gdn_idxbf16,
    "qwen3_8_flash_next_nvfp4_gdn_hcbf16": qwen3_8_flash_next_nvfp4_gdn_hcbf16,
    "qwen3_8_flash_next_nvfp4_gdnbf16": qwen3_8_flash_next_nvfp4_gdnbf16,
    "qwen3_8_flash_next_nvfp4_gdn_idxhcbf16": qwen3_8_flash_next_nvfp4_gdn_idxhcbf16,
    "qwen3_8_flash_next_nvfp4_allbf16": qwen3_8_flash_next_nvfp4_allbf16,
    "qwen3_8_flash_next_nvfp4_idxhcbf16_gdnq8": qwen3_8_flash_next_nvfp4_idxhcbf16_gdnq8,
    "qwen3_8_flash_next_nvfp4_idxbf16_gdnq8": qwen3_8_flash_next_nvfp4_idxbf16_gdnq8,
}
