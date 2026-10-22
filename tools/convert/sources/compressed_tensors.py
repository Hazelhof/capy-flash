"""Interpret the current compressed-tensors FP8/NVFP4/pack-quantized-int fields and scale semantics.

The matrix resolver also accepts direct tensors. Resolution remains lazy so a
recipe can replace an unused checkpoint source before its weights are inspected.
"""

from __future__ import annotations

from math import prod
import struct

import torch

from tools.artifact.codecs.fp8_row import validate_fp8_row_words
from tools.artifact.formats import valid_positive_fp32_word
from .logical import EncodedRows, LogicalSource
from .modelopt import MXFP8, is_mxfp8, mxfp8_source
from .safetensors import SafetensorsSource, tensor_source


def _divisor_word(store: SafetensorsSource, name: str) -> bytes:
    info = store.describe(name)
    if info.dtype != "F32" or prod(info.shape) != 1:
        raise ValueError(f"{name}: expected a source FP32 scalar")
    tensor = store.read_flat(name)
    raw = tensor.view(torch.uint8).numpy().tobytes()
    if not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
        raise ValueError(f"{name}: divisor must be finite and positive")
    return raw


def compressed_matrix_source(
    store: SafetensorsSource, prefix: str, shape: tuple[int, int], format: str
) -> LogicalSource:
    """Interpret the current compressed-tensors NVFP4 or per-row FP8 representation."""
    if format not in ("nvfp4", "fp8_e4m3fn_row_bf16"):
        raise ValueError(f"unsupported encoded source format {format}")
    n, k = shape
    if format == "nvfp4" and k % 16:
        raise ValueError(f"{prefix}: NVFP4 source K must be divisible by 16")

    def signature(name: str, expected: tuple[int, ...], dtype: str) -> None:
        info = store.describe(name)
        if info.shape != expected or info.dtype != dtype:
            raise ValueError(f"{name}: expected {dtype}{expected}, got {info.dtype}{info.shape}")

    def divisor(suffix: str) -> bytes:
        return _divisor_word(store, f"{prefix}.{suffix}")

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        if format == "nvfp4":
            packed, scale = f"{prefix}.weight_packed", f"{prefix}.weight_scale"
            signature(packed, (n, k // 2), "U8")
            signature(scale, (n, k // 16), "F8_E4M3")
            codes = store.read_flat(packed, begin * (k // 2), end * (k // 2)).reshape(end - begin, k // 2)
            scales = (
                store.read_flat(scale, begin * (k // 16), end * (k // 16))
                .view(torch.uint8)
                .reshape(end - begin, k // 16)
            )
            if bool((scales > 0x7E).any()):
                raise ValueError(f"{scale}: expected nonnegative finite E4M3FN scales")
            return EncodedRows(format, codes, scales, divisor("weight_global_scale"))
        weight, scale = f"{prefix}.weight", f"{prefix}.weight_scale"
        signature(weight, shape, "F8_E4M3")
        info = store.describe(scale)
        if info.dtype != "BF16" or prod(info.shape) != n:
            raise ValueError(f"{scale}: expected one BF16 scale per row")
        codes = store.read_flat(weight, begin * k, end * k).view(torch.uint8).reshape(end - begin, k)
        scales = store.read_flat(scale, begin, end)
        validate_fp8_row_words(codes, scales)
        return EncodedRows(format, codes, scales)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        if format == "fp8_e4m3fn_row_bf16":
            values = words.codes.view(torch.float8_e4m3fn).float() * words.scales.float()[:, None]
        else:
            codes = torch.stack((words.codes & 15, words.codes >> 4), dim=-1).reshape(last - first, k)
            values_table = torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ]
            )
            values = values_table[codes.long()]
            scales = words.scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
            values = values * scales / struct.unpack("<f", words.weight_divisor)[0]
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(
        shape,
        f"{store.path}:{prefix} ({format})",
        read,
        encoded,
        (lambda: divisor("weight_global_scale")) if format == "nvfp4" else None,
        (lambda: divisor("input_global_scale")) if format == "nvfp4" else None,
    )


# compressed-tensors "pack-quantized" symmetric integer weights (ISTA-DASLab GSQ checkpoints):
# ``weight_packed`` I32[N, K*bits/32] is one dense little-endian bit stream per row, element i at
# bits [bits*i, bits*i+bits) holding the unsigned code q + 2^(bits-1); ``weight_scale`` BF16[N, K/G]
# is the per-group multiplier (w = q * scale, negative for about half of the 3-bit groups);
# ``weight_shape`` I64[2] = [N, K]. The registered row-split formats keep the same signed codes and
# convert each scale to binary16.
PACKED_INT_FORMATS = {"q3_g128_fp16": (3, 128), "q4_g64_fp16": (4, 64)}

# BF16 -> binary16 multipliers: the only rounding a verbatim repack performs. Finite BF16 values
# below the binary16 normal range lose mantissa bits; the conversion is accepted when every
# multiplier stays within this absolute error (GSQ-3 27B: 218 of 240 271 360 inexact, max 2.98e-8).
PACKED_SCALE_ABS_ERROR = 2.0**-25


def packed_int_format(store: SafetensorsSource, prefix: str, shape: tuple[int, int]) -> str | None:
    """Name the registered format of a pack-quantized integer matrix, or None."""
    packed, scale = f"{prefix}.weight_packed", f"{prefix}.weight_scale"
    if not store.has(packed) or not store.has(scale) or store.describe(packed).dtype != "I32":
        return None
    n, k = shape
    words, groups = store.describe(packed).shape, store.describe(scale).shape
    for name, (bits, group) in PACKED_INT_FORMATS.items():
        if k % group == 0 and (k * bits) % 32 == 0 and tuple(words) == (n, k * bits // 32) and tuple(groups) == (n, k // group):
            return name
    raise ValueError(f"{prefix}: pack-quantized geometry {words}/{groups} matches no registered format")


def unpack_int_stream(words: torch.Tensor, bits: int, k: int) -> torch.Tensor:
    """Signed codes [rows, k] from I32[rows, k*bits/32] dense little-endian bit streams."""
    rows = words.shape[0]
    raw = words.contiguous().view(torch.uint8).reshape(rows, k * bits // 8).to(torch.int32)
    if bits == 4:
        unsigned = torch.stack((raw & 0xF, raw >> 4), dim=-1).reshape(rows, k)
    elif bits == 3:
        triples = raw.reshape(rows, k // 8, 3)
        stream = triples[..., 0] | (triples[..., 1] << 8) | (triples[..., 2] << 16)
        shifts = 3 * torch.arange(8, dtype=torch.int32)
        unsigned = ((stream.unsqueeze(-1) >> shifts) & 7).reshape(rows, k)
    else:
        raise ValueError(f"unsupported packed width {bits}")
    return (unsigned - (1 << (bits - 1))).to(torch.int8)


def packed_scales_fp16(scales: torch.Tensor, label: str, *, signed: bool = False) -> torch.Tensor:
    """BF16 multipliers -> binary16, rejecting any conversion beyond PACKED_SCALE_ABS_ERROR.

    ``signed`` admits negative multipliers (q3_g128_fp16: GSQ uses the sign of the scale to put the
    extra level of the asymmetric [-4, 3] grid on either side; about half of its groups are
    negative). Grouped formats with a symmetric use keep the nonnegative rule.
    """
    wide = scales.float()
    if not bool(torch.isfinite(wide).all()) or (not signed and bool((wide < 0).any())):
        raise ValueError(f"{label}: multipliers must be finite{'' if signed else ' and nonnegative'}")
    half = wide.to(torch.float16)
    if not bool(torch.isfinite(half).all()):
        raise ValueError(f"{label}: a multiplier overflows binary16")
    error = (half.float() - wide).abs()
    if bool((error > PACKED_SCALE_ABS_ERROR).any()):
        raise ValueError(f"{label}: binary16 multiplier error {float(error.max())} exceeds 2^-25")
    return half


def packed_int_matrix_source(
    store: SafetensorsSource, prefix: str, shape: tuple[int, int], format: str
) -> LogicalSource:
    """Signed codes and binary16 multipliers of a pack-quantized integer matrix, read verbatim."""
    bits, group = PACKED_INT_FORMATS[format]
    n, k = shape
    packed, scale, stored_shape = f"{prefix}.weight_packed", f"{prefix}.weight_scale", f"{prefix}.weight_shape"
    if store.has(stored_shape):
        recorded = tuple(int(v) for v in store.read_flat(stored_shape).tolist())
        if recorded != (n, k):
            raise ValueError(f"{prefix}: weight_shape {recorded} differs from {shape}")
    if store.describe(scale).dtype != "BF16":
        raise ValueError(f"{scale}: expected BF16 multipliers")
    words_per_row, groups_per_row = k * bits // 32, k // group

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        words = store.read_flat(packed, begin * words_per_row, end * words_per_row).reshape(end - begin, words_per_row)
        codes = unpack_int_stream(words, bits, k).reshape(end - begin, groups_per_row, group)
        scales = store.read_flat(scale, begin * groups_per_row, end * groups_per_row).reshape(end - begin, groups_per_row)
        return EncodedRows(format, codes, packed_scales_fp16(scales, scale, signed=bits == 3))

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        rows = encoded(first, last)
        values = rows.codes.float() * rows.scales.float().unsqueeze(-1)
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(shape, f"{store.path}:{prefix} ({format})", read, encoded)


def matrix_source(
    store: SafetensorsSource,
    name: str,
    shape: tuple[int, int],
    format: str | None = None,
) -> LogicalSource:
    """Resolve the selected matrix's encoding lazily, after recipe source overrides.

    An E4M3FN ``weight`` with a U8 ``weight_scale`` per 32 columns is a ModelOpt MXFP8 matrix.
    """
    prefix = name.removesuffix(".weight")
    resolved: LogicalSource | None = None

    def resolve() -> LogicalSource:
        nonlocal resolved
        if resolved is None:
            actual = format
            if actual is None:
                actual = packed_int_format(store, prefix, shape)
            if actual is None and store.has(prefix + ".weight_packed"):
                actual = "nvfp4"
            if actual is None and is_mxfp8(store, prefix):
                actual = MXFP8
            if actual is None and store.describe(name).dtype == "F8_E4M3":
                actual = "fp8_e4m3fn_row_bf16"
            if actual is None:
                resolved = tensor_source(store, name, shape)
            elif actual == MXFP8:
                resolved = mxfp8_source(store, prefix, shape)
            elif actual in PACKED_INT_FORMATS:
                resolved = packed_int_matrix_source(store, prefix, shape, actual)
            else:
                resolved = compressed_matrix_source(store, prefix, shape, actual)
        return resolved

    def encoded(begin: int, end: int) -> EncodedRows:
        reader = resolve().read_encoded
        if reader is None:
            raise ValueError(f"{name}: selected source does not provide encoded rows")
        return reader(begin, end)

    def divisor(which: str) -> bytes:
        read = getattr(resolve(), which)
        if read is None:
            raise ValueError(f"{name}: selected source does not provide {which}")
        return read()

    return LogicalSource(
        shape,
        f"{store.path}:{name}",
        lambda begin, end: resolve().values(begin, end),
        encoded,
        lambda: divisor("weight_divisor"),
        lambda: divisor("input_divisor"),
    )
