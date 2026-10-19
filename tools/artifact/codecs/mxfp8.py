"""Verbatim E4M3FN code words and UE8M0 group scales in block_scale_k32_v1 layout.

value(n, k) = e4m3fn(code[n, k]) * 2 ** (scale[n, k // 32] - 127), exact in FP32.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..layouts import block_scale_k32_geometry
from ._tensor_bytes import (
    Payload,
    _exact_uint8_matrix,
    _payload_length,
    _payload_tensor,
)

FORMAT = "mxfp8_e4m3fn_g32_ue8m0"


def validate_mxfp8_words(codes: torch.Tensor, scales: torch.Tensor) -> None:
    if bool(((codes & 0x7F) == 0x7F).any()):
        raise ValueError("MXFP8 codes must be finite E4M3FN words")
    if bool((scales == 0xFF).any()):
        raise ValueError("MXFP8 scales must be finite UE8M0 words (255 is NaN)")


def encode_mxfp8(code_words: torch.Tensor, scale_words: torch.Tensor, shape: Sequence[int]) -> bytes:
    """Copy exact E4M3FN code words [N, K] and UE8M0 scale words [N, K/32] into their planes."""

    geometry = block_scale_k32_geometry(FORMAT, shape)
    codes = _exact_uint8_matrix(code_words, (geometry.n, geometry.k), "MXFP8 codes")
    scales = _exact_uint8_matrix(scale_words, (geometry.n, geometry.groups_per_row), "MXFP8 scales")
    validate_mxfp8_words(codes, scales)
    payload = bytearray(geometry.payload_bytes)
    payload[: geometry.code_plane_bytes] = codes.numpy().tobytes()
    begin = geometry.scale_plane_offset
    payload[begin : begin + geometry.scale_plane_bytes] = scales.numpy().tobytes()
    return bytes(payload)


def decode_mxfp8_words(payload: Payload, shape: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact code words [N, K] and scale words [N, K/32] of a payload."""

    geometry = block_scale_k32_geometry(FORMAT, shape)
    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError(f"MXFP8 payload has {_payload_length(payload)} bytes, expected {geometry.payload_bytes}")
    raw = _payload_tensor(payload, torch.device("cpu"))
    codes = raw[: geometry.code_plane_bytes].clone().reshape(geometry.n, geometry.k)
    begin = geometry.scale_plane_offset
    scales = raw[begin : begin + geometry.scale_plane_bytes].clone().reshape(geometry.n, geometry.groups_per_row)
    validate_mxfp8_words(codes, scales)
    return codes, scales


def dequantize_mxfp8(payload: Payload, shape: Sequence[int], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Reconstruct the MXFP8 matrix from its stored words (exact in FP32)."""

    codes, scales = decode_mxfp8_words(payload, shape)
    # 2 ** (s - 127) is the FP32 word with biased exponent s (s = 0: the subnormal 2 ** -127).
    words = scales.to(torch.int32)
    multipliers = torch.where(words > 0, words << 23, torch.full_like(words, 1 << 22)).view(torch.float32)
    values = codes.view(torch.float8_e4m3fn).float() * multipliers.repeat_interleave(32, dim=1)
    return values.to(dtype)
