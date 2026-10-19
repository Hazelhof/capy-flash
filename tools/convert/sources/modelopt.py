"""Interpret NVIDIA ModelOpt NVFP4, MXFP8, row-NVFP4, block-FP8 and per-tensor FP8 checkpoint fields.

ModelOpt stores multipliers where the v3 formats store divisors or row multipliers:

- NVFP4 ``weight`` (U8 packed E2M1, low nibble first), ``weight_scale`` (E4M3FN per 16 columns),
  ``weight_scale_2`` (FP32 multiplier) and ``input_scale`` (FP32 activation multiplier). The v3
  weight divisor is ``1 / weight_scale_2``, rounded once to FP32; codes and block scales are
  preserved bit-exactly. The block scales may instead be CSF-compressed
  (``weight_scale.nvfp4_csf_fixed`` + ``.nvfp4_csf_exceptions``, see csf.py); they decode
  losslessly to the same E4M3FN bytes.
- MXFP8 ``weight`` (E4M3FN) with ``weight_scale`` (U8 UE8M0 exponent per 32 columns): value
  ``e4m3 * 2^(scale - 127)``. Codes and scales are copied verbatim into ``mxfp8_e4m3fn_g32_ue8m0``.
- Row NVFP4 tables (the n-gram embedding shards): ``weight`` U8 ``[R, W/2]``, ``weight_scale``
  E4M3FN ``[R, W/16]`` per shard and one FP32 ``weight_scale_2`` multiplier G for all shards:
  value ``e2m1 * e4m3 * G``. Codes and scales are copied verbatim into ``nvfp4_row_e4m3_g16``;
  G is a separate one-element FP32 parameter.
- Block FP8 ``weight`` (E4M3FN) with ``weight_scale_inv`` (BF16 multiplier per 128x128 tile).
  It has no v3 encoded counterpart and is exposed as logical values only.
- Per-tensor FP8 ``weight`` (E4M3FN) with one BF16 ``weight_scale`` multiplier. It maps exactly to
  the row-scaled FP8 format with that multiplier on every row.
"""

from __future__ import annotations

from bisect import bisect_right
from functools import cache
from math import prod
import struct

import torch

from tools.artifact.codecs.fp8_row import validate_fp8_row_words
from tools.artifact.formats import valid_positive_fp32_word
from .csf import EXCEPTIONS_SUFFIX, FIXED_SUFFIX, SLAB_ROWS, csf_shape, decode_csf_slabs, exception_words
from .logical import EncodedRows, LogicalSource
from .safetensors import SafetensorsSource

MXFP8 = "mxfp8_e4m3fn_g32_ue8m0"
NVFP4_ROW = "nvfp4_row_e4m3_g16"

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _signature(store: SafetensorsSource, name: str, shape: tuple[int, ...], dtype: str) -> None:
    info = store.describe(name)
    if tuple(info.shape) != tuple(shape) or info.dtype != dtype:
        raise ValueError(f"{name}: expected {dtype}{shape}, got {info.dtype}{info.shape}")


def _scalar_f32(store: SafetensorsSource, name: str) -> float:
    info = store.describe(name)
    if info.dtype != "F32" or prod(info.shape) != 1:
        raise ValueError(f"{name}: expected one FP32 value")
    return float(store.read_flat(name)[0])


def reciprocal_word(value: float, label: str) -> bytes:
    """FP32 word of ``1 / value``; the only rounding a ModelOpt multiplier import performs."""
    if not value > 0 or value != value or value == float("inf"):
        raise ValueError(f"{label}: multiplier must be finite and positive")
    word = struct.pack("<f", 1.0 / value)
    if not valid_positive_fp32_word(struct.unpack("<I", word)[0]):
        raise ValueError(f"{label}: reciprocal is not a positive finite FP32 value")
    return word


def decode_nvfp4(codes: torch.Tensor, scales: torch.Tensor, divisor: bytes) -> torch.Tensor:
    """Exact represented values ``e2m1 * e4m3 / d_w`` of packed rows."""
    rows = codes.shape[0]
    words = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(rows, -1)
    values = _E2M1[words.long()]
    block = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return values * block / struct.unpack("<f", divisor)[0]


def csf_scale_rows(store: SafetensorsSource, name: str, shape: tuple[int, int], begin: int, end: int) -> torch.Tensor:
    """Rows ``[begin, end)`` (U8) of the CSF-compressed E4M3FN scale matrix ``name`` of ``shape``.

    Only the 16-row slabs covering the rows are read; the exception stream is small and read whole.
    """
    fixed, exceptions = name + FIXED_SUFFIX, name + EXCEPTIONS_SUFFIX
    info = store.describe(fixed)
    if info.dtype != "U8" or csf_shape(tuple(info.shape)) != tuple(shape):
        raise ValueError(f"{fixed}: expected a U8 CSF stream of a {shape} scale matrix, got {info.dtype}{info.shape}")
    words = store.describe(exceptions)
    if words.dtype != "U32" or len(words.shape) != 1:
        raise ValueError(f"{exceptions}: expected one U32 vector, got {words.dtype}{words.shape}")
    rows, cols = shape
    width = info.shape[1]
    first, last = begin // SLAB_ROWS, -(-end // SLAB_ROWS)
    slabs = store.read_flat(fixed, first * width, last * width)
    positions, values = exception_words(store.read_flat(exceptions), rows, cols)
    decoded = decode_csf_slabs(slabs, positions, values, first_slab=first, cols=cols)
    return decoded[begin - first * SLAB_ROWS : end - first * SLAB_ROWS].contiguous()


def nvfp4_source(store: SafetensorsSource, prefix: str, shape: tuple[int, int]) -> LogicalSource:
    """One ModelOpt NVFP4 matrix ``prefix.weight`` with logical shape ``[N,K]``."""
    n, k = shape
    if k % 16:
        raise ValueError(f"{prefix}: NVFP4 K must be divisible by 16")

    # Each chunk of the matrix names its divisor; read the scalar once, not once per chunk.
    @cache
    def divisor() -> bytes:
        return reciprocal_word(_scalar_f32(store, prefix + ".weight_scale_2"), prefix)

    @cache
    def input_divisor() -> bytes:
        return reciprocal_word(_scalar_f32(store, prefix + ".input_scale"), prefix)

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        _signature(store, prefix + ".weight", (n, k // 2), "U8")
        codes = store.read_flat(prefix + ".weight", begin * (k // 2), end * (k // 2))
        if store.has(prefix + ".weight_scale" + FIXED_SUFFIX):
            scales = csf_scale_rows(store, prefix + ".weight_scale", (n, k // 16), begin, end)
        else:
            _signature(store, prefix + ".weight_scale", (n, k // 16), "F8_E4M3")
            scales = store.read_flat(prefix + ".weight_scale", begin * (k // 16), end * (k // 16))
            scales = scales.view(torch.uint8).reshape(end - begin, k // 16)
        if bool((scales > 0x7E).any()):
            raise ValueError(f"{prefix}.weight_scale: expected nonnegative finite E4M3FN")
        return EncodedRows("nvfp4", codes.reshape(end - begin, k // 2), scales, divisor())

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        values = decode_nvfp4(words.codes, words.scales, words.weight_divisor)
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(shape, f"{store.path}:{prefix} (modelopt nvfp4)", read, encoded, divisor, input_divisor)


def fp8_block_source(store: SafetensorsSource, prefix: str, shape: tuple[int, int], block: int = 128) -> LogicalSource:
    """ModelOpt/DeepSeek block FP8 ``weight * weight_scale_inv[tile]`` as logical values."""
    n, k = shape
    tiles = ((n + block - 1) // block, (k + block - 1) // block)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        _signature(store, prefix + ".weight", shape, "F8_E4M3")
        _signature(store, prefix + ".weight_scale_inv", tiles, "BF16")
        first, last = begin // k, (end + k - 1) // k
        codes = store.read_flat(prefix + ".weight", first * k, last * k).reshape(last - first, k)
        scale = store.read_flat(prefix + ".weight_scale_inv").reshape(tiles).float()
        rows = torch.arange(first, last) // block
        tile = scale[rows].repeat_interleave(block, dim=1)[:, :k]
        values = codes.view(torch.float8_e4m3fn).float() * tile
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(shape, f"{store.path}:{prefix} (block fp8)", read)


def fp8_tensor_rows_source(
    store: SafetensorsSource,
    shards: list[str],
    scale: str,
    shape: tuple[int, int],
) -> LogicalSource:
    """Row-concatenated per-tensor FP8 shards sharing one BF16 multiplier.

    Encoded rows carry the shared multiplier as every row scale, which is the exact
    ``fp8_e4m3fn_row_bf16`` representation of the same values.
    """
    n, k = shape
    counts = []
    for name in shards:
        info = store.describe(name)
        if info.dtype != "F8_E4M3" or len(info.shape) != 2 or info.shape[1] != k:
            raise ValueError(f"{name}: expected F8_E4M3 rows of width {k}")
        counts.append(info.shape[0])
    if sum(counts) != n:
        raise ValueError(f"{shards[0]}: shard rows {sum(counts)} differ from {n}")
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)

    def multiplier() -> torch.Tensor:
        info = store.describe(scale)
        if info.dtype != "BF16" or prod(info.shape) != 1:
            raise ValueError(f"{scale}: expected one BF16 multiplier")
        return store.read_flat(scale).reshape(1)

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{shards[0]}: invalid encoded rows [{begin},{end})")
        pieces = []
        for index, name in enumerate(shards):
            low, high = max(begin, starts[index]), min(end, starts[index + 1])
            if low < high:
                local = low - starts[index], high - starts[index]
                pieces.append(store.read_flat(name, local[0] * k, local[1] * k))
        codes = torch.cat(pieces).view(torch.uint8).reshape(end - begin, k)
        scales = multiplier().expand(end - begin).contiguous()
        validate_fp8_row_words(codes, scales)
        return EncodedRows("fp8_e4m3fn_row_bf16", codes, scales)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        values = words.codes.view(torch.float8_e4m3fn).float() * words.scales.float()[:, None]
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(shape, f"{store.path}:{shards[0]}+{len(shards) - 1} (fp8)", read, encoded)


def decode_mxfp8(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Exact represented values ``e4m3 * 2^(ue8m0 - 127)`` of MXFP8 rows (FP32)."""
    exponent = scales.to(torch.int32).repeat_interleave(32, dim=1) - 127
    return torch.ldexp(codes.view(torch.float8_e4m3fn).float(), exponent.float())


def mxfp8_source(store: SafetensorsSource, prefix: str, shape: tuple[int, int]) -> LogicalSource:
    """ModelOpt MXFP8 matrix ``prefix.weight`` (E4M3FN) with UE8M0 ``prefix.weight_scale`` per 32 columns.

    Encoded rows are the checkpoint bytes: codes ``[rows, K]`` and exponents ``[rows, K/32]``.
    """
    n, k = shape
    if k % 32:
        raise ValueError(f"{prefix}: MXFP8 K must be divisible by 32")

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        _signature(store, prefix + ".weight", (n, k), "F8_E4M3")
        _signature(store, prefix + ".weight_scale", (n, k // 32), "U8")
        codes = store.read_flat(prefix + ".weight", begin * k, end * k).view(torch.uint8).reshape(end - begin, k)
        scales = store.read_flat(prefix + ".weight_scale", begin * (k // 32), end * (k // 32))
        scales = scales.reshape(end - begin, k // 32)
        if bool((scales == 0xFF).any()):
            raise ValueError(f"{prefix}.weight_scale: UE8M0 exponent 255 is not a finite scale")
        if bool(((codes & 0x7F) == 0x7F).any()):
            raise ValueError(f"{prefix}.weight: NaN E4M3FN code")
        return EncodedRows(MXFP8, codes, scales)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        values = decode_mxfp8(words.codes, words.scales)
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(shape, f"{store.path}:{prefix} (modelopt mxfp8)", read, encoded)


def is_mxfp8(store: SafetensorsSource, prefix: str) -> bool:
    """Whether ``prefix.weight`` is an MXFP8 matrix (E4M3FN codes, U8 exponents per 32 columns)."""
    weight, scale = prefix + ".weight", prefix + ".weight_scale"
    if not (store.has(weight) and store.has(scale)):
        return False
    info, exponents = store.describe(weight), store.describe(scale)
    return (
        info.dtype == "F8_E4M3"
        and exponents.dtype == "U8"
        and len(info.shape) == 2
        and info.shape[1] % 32 == 0
        and tuple(exponents.shape) == (info.shape[0], info.shape[1] // 32)
    )


def nvfp4_row_global_scale(store: SafetensorsSource, name: str) -> LogicalSource:
    """The row-NVFP4 table multiplier G (``weight_scale_2``) as one FP32 value, bits preserved."""

    def read(begin: int, end: int) -> torch.Tensor:
        info = store.describe(name)
        if info.dtype != "F32" or prod(info.shape) != 1:
            raise ValueError(f"{name}: expected one FP32 multiplier")
        value = store.read_flat(name).reshape(1)
        if not bool(torch.isfinite(value).all()) or not float(value[0]) > 0:
            raise ValueError(f"{name}: multiplier must be finite and positive")
        return value[begin:end]

    return LogicalSource((1,), f"{store.path}:{name} (nvfp4 row global scale)", read)


def nvfp4_row_table_source(
    store: SafetensorsSource,
    shards: list[str],
    global_scale: str,
    shape: tuple[int, int],
) -> LogicalSource:
    """Row-concatenated row-NVFP4 shards ``<shard>.weight``/``<shard>.weight_scale`` sharing G.

    Encoded rows are the checkpoint bytes: codes ``[rows, W/2]`` (low nibble = even element) and
    E4M3FN scales ``[rows, W/16]``. Values are ``e2m1 * e4m3 * G`` in FP32.
    """
    n, width = shape
    if width % 16:
        raise ValueError(f"{shards[0]}: row-NVFP4 width must be divisible by 16")
    counts = []
    for name in shards:
        codes, scales = store.describe(name + ".weight"), store.describe(name + ".weight_scale")
        if codes.dtype != "U8" or len(codes.shape) != 2 or codes.shape[1] != width // 2:
            raise ValueError(f"{name}.weight: expected U8 rows of {width // 2} bytes")
        if scales.dtype != "F8_E4M3" or tuple(scales.shape) != (codes.shape[0], width // 16):
            raise ValueError(f"{name}.weight_scale: expected F8_E4M3 [{codes.shape[0]}, {width // 16}]")
        counts.append(codes.shape[0])
    if sum(counts) != n:
        raise ValueError(f"{shards[0]}: shard rows {sum(counts)} differ from {n}")
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    multiplier = nvfp4_row_global_scale(store, global_scale)

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{shards[0]}: invalid encoded rows [{begin},{end})")
        codes, scales = [], []
        first = bisect_right(starts, begin) - 1
        for index in range(first, len(shards)):
            low, high = max(begin, starts[index]), min(end, starts[index + 1])
            if low >= high:
                break
            local = low - starts[index], high - starts[index]
            name = shards[index]
            codes.append(store.read_flat(name + ".weight", local[0] * (width // 2), local[1] * (width // 2)))
            scales.append(store.read_flat(name + ".weight_scale", local[0] * (width // 16), local[1] * (width // 16)))
        codes = (codes[0] if len(codes) == 1 else torch.cat(codes)).reshape(end - begin, width // 2)
        scales = (scales[0] if len(scales) == 1 else torch.cat(scales)).view(torch.uint8)
        scales = scales.reshape(end - begin, width // 16)
        if bool(((scales & 0x7F) == 0x7F).any()):
            raise ValueError(f"{shards[0]}: NaN E4M3FN row scale")
        return EncodedRows(NVFP4_ROW, codes, scales)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // width, (end + width - 1) // width
        words = encoded(first, last)
        nibbles = torch.stack((words.codes & 15, words.codes >> 4), dim=-1).reshape(last - first, width)
        block = words.scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
        values = _E2M1[nibbles.long()] * block * multiplier.values()[0]
        return values.reshape(-1)[begin - first * width : end - first * width]

    return LogicalSource(shape, f"{store.path}:{shards[0]}+{len(shards) - 1} (nvfp4 rows)", read, encoded)
