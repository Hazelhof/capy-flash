"""Qwen4Exp checkpoint -> converter -> C++ loader, end to end, on a tiny synthetic checkpoint.

Writes the ModelOpt-layout fixture of tests/convert/qwen4_exp_fixture.py, converts it with both
Flash-Next recipes (the Q8-dense one through the CLI), and runs the C++ loading test on each
artifact with an expectation file computed from the checkpoint itself: the hyper-parameters, the
PLE hash constants the checkpoint stores, the exact bytes an expert cache slot must hold (codes,
swizzled scales and divisor words of every routed expert, rebuilt from the source tensors with the
artifact codecs, not the converter) and n-gram table rows.

Usage: qwen4_load_interop.py EXECUTABLE [--keep DIR]. Exit 77 when torch/safetensors are missing.
"""

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

try:
    import torch
    import safetensors  # noqa: F401
except ImportError:
    print("SKIP: qwen4_load_interop needs torch and safetensors")
    sys.exit(77)

from tests.convert import qwen4_exp_fixture as fixture  # noqa: E402
from tools.artifact.codecs.nvfp4 import swizzle_nvfp4_scales  # noqa: E402
from tools.convert.official_recipes import qwen3_8_flash_next_nvfp4_dense  # noqa: E402
from tools.convert.pipeline import convert  # noqa: E402
from tools.convert.qwen4_exp import build_model  # noqa: E402
from tools.convert.recipe import Recipe  # noqa: E402
from tools.convert.sources.safetensors import SafetensorsSource  # noqa: E402


def slot_bytes(tensors: dict, prefix: str) -> bytes:
    """An expert's slot: per bank (gate|up, down) its codes, swizzled scales, divisor words."""
    out = b""
    for roles in (("gate", "up"), ("down",)):
        names = [f"{prefix}{role}_proj" for role in roles]
        codes = torch.cat([tensors[n + ".weight"] for n in names])
        scales = torch.cat([tensors[n + ".weight_scale"].view(torch.uint8) for n in names])
        rows, k = codes.shape[0], codes.shape[1] * 2
        out += codes.numpy().tobytes()
        out += swizzle_nvfp4_scales(scales, (rows, k)).numpy().tobytes()
        out += b"".join(fixture.divisor_word(tensors[n + ".weight_scale_2"]) for n in names)
    return out


def expectations(tensors: dict, directory: Path, dense: bool) -> Path:
    text = fixture.config()["text_config"]
    constants = fixture.ple_constants(text)
    lm = "model.language_model."
    emb = f"{lm}layers.0.ple.ple_embedding."
    stored = {
        "multipliers": [int(v) & ((1 << 64) - 1) for v in tensors[emb + "layer_multipliers"].tolist()],
        "moduli": tensors[emb + "ngram_heads_vocab_sizes"].tolist(),
        "offsets": tensors[emb + "ngram_heads_offsets"].tolist(),
    }
    slots = []
    for layer in range(len(fixture.LAYER_TYPES)):
        for expert in range(fixture.E):
            name = f"slot_text_{layer}_{expert}.bin"
            (directory / name).write_bytes(
                slot_bytes(tensors, f"{lm}layers.{layer}.mlp.experts.{expert}.")
            )
            slots.append({"component": "text", "layer": layer, "expert": expert, "file": name})
    table = torch.cat(
        [tensors[f"{emb}ngram_embedding.shard_{s}.weight"].view(torch.uint8) for s in (0, 1)]
    )
    scale = tensors[emb + "ngram_embedding.weight_scale"].view(torch.uint8).numpy().tobytes()
    rows = [
        {"row": r, "codes": table[r].numpy().tobytes().hex(), "scale": scale.hex()}
        for r in (0, constants["rows"] // 2 - 1, constants["rows"] // 2, constants["rows"] - 1)
    ]
    nvfp4_or_q8 = "nvfp4" if dense else "q8_g32_fp16"
    value = {
        "hidden": fixture.H,
        "layers": len(fixture.LAYER_TYPES),
        "vocab": fixture.VOCAB,
        "experts": fixture.E,
        "experts_per_token": text["num_experts_per_tok"],
        "moe_intermediate": fixture.IR,
        "shared_intermediate": fixture.SHARED,
        "hc_count": fixture.HC,
        "hc_lowrank": fixture.LOWRANK,
        "layer_types": fixture.LAYER_TYPES,
        "qsa": {
            "heads": text["indexer_n_heads"],
            "head_dim": text["indexer_head_dim"],
            "budget": text["indexer_budget"],
            "compress_ratio": text["indexer_compress_ratio"],
        },
        "ple": {
            "layer": fixture.PLE_LAYER_IDS[0] - 1,
            "rows": constants["rows"],
            "row_width": fixture.PLE_DIM // ((fixture.NGRAM - 1) * fixture.HEADS_PER_NGRAM),
            **stored,
        },
        "formats": {
            "text/output_head": "nvfp4" if dense else "q6_g64_fp16",
            "text/token_embedding": "q8_g32_fp16",
            "text/layers/1/attention/query": nvfp4_or_q8,
            "text/layers/0/gdn/value": nvfp4_or_q8,
            "text/layers/0/moe/shared/down": nvfp4_or_q8,
            "text/layers/1/attention/indexer_key": "q8_g32_fp16",
            "text/layers/0/ple/table": "fp8_e4m3fn_row_bf16",
            "text/layers/0/moe/router": "bf16",
        },
        "expert_slots": slots,
        "table_rows": rows,
    }
    # The loader derives these from seed/base/divisor; the checkpoint states them.
    assert stored["moduli"] == constants["moduli"] and stored["offsets"] == constants["offsets"]
    path = directory / ("expect_dense.json" if dense else "expect_q8.json")
    path.write_text(json.dumps(value, indent=1))
    return path


def main() -> int:
    executable = sys.argv[1]
    keep = Path(sys.argv[sys.argv.index("--keep") + 1]) if "--keep" in sys.argv else None
    with tempfile.TemporaryDirectory(prefix="ninfer-qwen4-load-") as temporary:
        work = keep or Path(temporary)
        work.mkdir(parents=True, exist_ok=True)
        source = work / "checkpoint"
        tensors = fixture.write_checkpoint(source)
        q8 = work / "flash_next_q8.ninfer"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.convert",
                "--model",
                str(source),
                "--recipe",
                "qwen3_8_flash_next_nvfp4",
                "--components",
                "text,mtp",
                "--device",
                "cpu",
                "--rows-per-chunk",
                "128",
                "--out",
                str(q8),
            ],
            check=True,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            stdout=subprocess.DEVNULL,
        )
        dense = work / "flash_next_dense.ninfer"
        with SafetensorsSource(source) as base:
            model = build_model(base, components=("text", "mtp"))
            recipe = Recipe(model)
            qwen3_8_flash_next_nvfp4_dense(model, recipe, {"base": base})
            convert(model, recipe, dense, device="cpu", rows_per_chunk=128, max_file_bytes=200_000)
        runs = [
            (q8, expectations(tensors, work, dense=False), []),
            (q8, work / "expect_q8.json", ["--embedding-host"]),
            (dense, expectations(tensors, work, dense=True), []),
        ]
        for artifact, expect, extra in runs:
            command = [executable, "--artifact", str(artifact), "--expect", str(expect), "--mtp"]
            result = subprocess.run(command + extra)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
