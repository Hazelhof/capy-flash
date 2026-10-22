"""Closed registry of persistent NInfer tensor numeric formats."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from types import MappingProxyType
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class DirectFormat:
    """One fixed-width word per logical tensor element."""

    name: str
    word_bytes: int


@dataclass(frozen=True, slots=True)
class QuantFormat:
    """Signed grouped codes with one binary16 multiplier per group."""

    name: str
    bits: int
    group_size: int
    qmin: int
    qmax: int


@dataclass(frozen=True, slots=True)
class Nvfp4Format:
    """E2M1 weights with one E4M3FN scale word per K-axis group."""

    name: str
    group_size: int


@dataclass(frozen=True, slots=True)
class Fp8RowFormat:
    """E4M3FN weights with one BF16 multiplier per logical row."""

    name: str


@dataclass(frozen=True, slots=True)
class Mxfp8Format:
    """E4M3FN weights with one UE8M0 power-of-two scale word per K-axis group (OCP MX)."""

    name: str
    group_size: int


@dataclass(frozen=True, slots=True)
class Nvfp4RowFormat:
    """E2M1 weights with one E4M3FN scale word per row group of 16 elements.

    The tensor-wide FP32 multiplier is a separate logical parameter, not part of the payload.
    """

    name: str
    group_size: int


@dataclass(frozen=True, slots=True)
class GgufFormat:
    """One ggml block type, stored as the exporter wrote its blocks: whole blocks per row.

    `ggml_type` is the block type's id in the GGUF tensor directory. A row of K elements is
    K / block_elements consecutive blocks of block_bytes each, so a row subset is a byte copy.
    Names, ids and sizes follow iamwavecut/ninfer-all (Apache-2.0), which stores the same blocks
    in the same container; the block definitions are ggml's (ggml-common.h, MIT).
    """

    name: str
    ggml_type: int
    block_elements: int
    block_bytes: int


NumericFormat: TypeAlias = (
    DirectFormat | QuantFormat | Nvfp4Format | Fp8RowFormat | Mxfp8Format | Nvfp4RowFormat | GgufFormat
)


BF16 = DirectFormat("bf16", 2)
FP32 = DirectFormat("fp32", 4)
INT32 = DirectFormat("int32", 4)

# v0.7 27b-16gb: symmetric 3-bit codes, one binary16 multiplier per 128 values (3.125 bpw), the
# representation of the ISTA-DASLab 3-bit GSQ checkpoints (and of roofkid/ninfer-4080 Q3G128_F16S).
Q3_G128_FP16 = QuantFormat("q3_g128_fp16", 3, 128, -4, 3)
Q4_G64_FP16 = QuantFormat("q4_g64_fp16", 4, 64, -8, 7)
Q5_G64_FP16 = QuantFormat("q5_g64_fp16", 5, 64, -16, 15)
Q6_G64_FP16 = QuantFormat("q6_g64_fp16", 6, 64, -32, 31)
Q8_G32_FP16 = QuantFormat("q8_g32_fp16", 8, 32, -127, 127)
NVFP4 = Nvfp4Format("nvfp4", 16)
FP8_E4M3FN_ROW_BF16 = Fp8RowFormat("fp8_e4m3fn_row_bf16")
MXFP8_E4M3FN_G32_UE8M0 = Mxfp8Format("mxfp8_e4m3fn_g32_ue8m0", 32)
NVFP4_ROW_E4M3_G16 = Nvfp4RowFormat("nvfp4_row_e4m3_g16", 16)

# ggml block types of the GSQ-RCO routed-expert banks (ggml-common.h), kept byte for byte.
GGUF_IQ3_XXS = GgufFormat("gguf_iq3_xxs", 18, 256, 98)
GGUF_IQ4_NL = GgufFormat("gguf_iq4_nl", 20, 32, 18)
GGUF_IQ3_S = GgufFormat("gguf_iq3_s", 21, 256, 110)
GGUF_IQ2_S = GgufFormat("gguf_iq2_s", 22, 256, 82)
GGUF_IQ4_XS = GgufFormat("gguf_iq4_xs", 23, 256, 136)
# Two-bit codes over {-1, 0, 1, 2} * d, 64 per block with one binary16 d (GSQ-RCO's Q2_0).
GGUF_Q2_0 = GgufFormat("gguf_q2_0", 42, 64, 18)


DIRECT_FORMATS = MappingProxyType({item.name: item for item in (BF16, FP32, INT32)})
QUANT_FORMATS = MappingProxyType({item.name: item for item in (Q3_G128_FP16, Q4_G64_FP16, Q5_G64_FP16, Q6_G64_FP16, Q8_G32_FP16)})
NVFP4_FORMATS = MappingProxyType({NVFP4.name: NVFP4})
FP8_ROW_FORMATS = MappingProxyType({FP8_E4M3FN_ROW_BF16.name: FP8_E4M3FN_ROW_BF16})
MXFP8_FORMATS = MappingProxyType({MXFP8_E4M3FN_G32_UE8M0.name: MXFP8_E4M3FN_G32_UE8M0})
NVFP4_ROW_FORMATS = MappingProxyType({NVFP4_ROW_E4M3_G16.name: NVFP4_ROW_E4M3_G16})
GGUF_FORMATS = MappingProxyType(
    {
        item.name: item
        for item in (GGUF_IQ3_XXS, GGUF_IQ4_NL, GGUF_IQ3_S, GGUF_IQ2_S, GGUF_IQ4_XS, GGUF_Q2_0)
    }
)
GGUF_FORMATS_BY_TYPE = MappingProxyType({item.ggml_type: item for item in GGUF_FORMATS.values()})
NUMERIC_FORMATS = MappingProxyType(
    {
        **DIRECT_FORMATS,
        **QUANT_FORMATS,
        **NVFP4_FORMATS,
        **FP8_ROW_FORMATS,
        **MXFP8_FORMATS,
        **NVFP4_ROW_FORMATS,
        **GGUF_FORMATS,
    }
)


_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def decode_e2m1_word(word: int) -> float:
    """Decode one exact four-bit E2M1 word, including signed zero."""

    if type(word) is not int or not 0 <= word <= 0xF:
        raise ValueError("E2M1 word must be an integer in [0, 15]")
    magnitude = _E2M1_MAGNITUDES[word & 0x7]
    return math.copysign(magnitude, -1.0 if word & 0x8 else 1.0)


def decode_e4m3fn_word(word: int) -> float:
    """Decode one exact eight-bit E4M3FN word."""

    if type(word) is not int or not 0 <= word <= 0xFF:
        raise ValueError("E4M3FN word must be an integer in [0, 255]")
    sign = -1.0 if word & 0x80 else 1.0
    exponent = (word >> 3) & 0xF
    fraction = word & 0x7
    if exponent == 0:
        if fraction == 0:
            return math.copysign(0.0, sign)
        return sign * fraction * (2.0**-9)
    if exponent == 0xF and fraction == 0x7:
        return math.copysign(math.nan, sign)
    return sign * (1.0 + fraction / 8.0) * (2.0 ** (exponent - 7))


def valid_nvfp4_scale_word(word: int) -> bool:
    """Return whether *word* is an admitted nonnegative finite E4M3FN scale."""

    return type(word) is int and 0 <= word <= 0xFF and word & 0x80 == 0 and word != 0x7F


def valid_fp8_weight_word(word: int) -> bool:
    """Return whether *word* is a finite E4M3FN weight code."""

    return type(word) is int and 0 <= word <= 0xFF and (word & 0x7F) != 0x7F


def valid_ue8m0_scale_word(word: int) -> bool:
    """Return whether *word* is a finite UE8M0 power-of-two scale (255 is the MX NaN)."""

    return type(word) is int and 0 <= word <= 0xFE


def decode_ue8m0_word(word: int) -> float:
    """Decode one exact UE8M0 scale word: 2 ** (word - 127)."""

    if not valid_ue8m0_scale_word(word):
        raise ValueError("UE8M0 word must be an integer in [0, 254]")
    return 2.0 ** (word - 127)


def valid_fp8_row_scale_word(word: int) -> bool:
    """Return whether *word* is a nonnegative finite BF16 multiplier."""

    if type(word) is not int or not 0 <= word <= 0xFFFF or word & 0x8000:
        return False
    value = struct.unpack("<f", struct.pack("<I", word << 16))[0]
    return math.isfinite(value)


def valid_positive_fp32_word(word: int) -> bool:
    """Return whether an IEEE binary32 word represents a finite positive value."""

    if type(word) is not int or not 0 <= word <= 0xFFFFFFFF:
        return False
    value = struct.unpack("<f", struct.pack("<I", word))[0]
    return math.isfinite(value) and value > 0.0


def get_format(name: str) -> NumericFormat:
    """Return the registered format named *name*."""

    try:
        return NUMERIC_FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown numeric format: {name!r}") from None


__all__ = [
    "BF16",
    "FP32",
    "INT32",
    "Q3_G128_FP16",
    "Q4_G64_FP16",
    "Q5_G64_FP16",
    "Q6_G64_FP16",
    "Q8_G32_FP16",
    "NVFP4",
    "FP8_E4M3FN_ROW_BF16",
    "MXFP8_E4M3FN_G32_UE8M0",
    "NVFP4_ROW_E4M3_G16",
    "GGUF_IQ3_XXS",
    "GGUF_IQ4_NL",
    "GGUF_IQ3_S",
    "GGUF_IQ2_S",
    "GGUF_IQ4_XS",
    "GGUF_Q2_0",
    "GGUF_FORMATS",
    "GGUF_FORMATS_BY_TYPE",
    "GgufFormat",
    "DIRECT_FORMATS",
    "QUANT_FORMATS",
    "NVFP4_FORMATS",
    "FP8_ROW_FORMATS",
    "MXFP8_FORMATS",
    "NVFP4_ROW_FORMATS",
    "NUMERIC_FORMATS",
    "DirectFormat",
    "QuantFormat",
    "Nvfp4Format",
    "Fp8RowFormat",
    "Mxfp8Format",
    "Nvfp4RowFormat",
    "NumericFormat",
    "get_format",
    "decode_e2m1_word",
    "decode_e4m3fn_word",
    "decode_ue8m0_word",
    "valid_fp8_row_scale_word",
    "valid_fp8_weight_word",
    "valid_nvfp4_scale_word",
    "valid_positive_fp32_word",
    "valid_ue8m0_scale_word",
]
