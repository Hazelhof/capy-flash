"""Verbatim E2M1 code bytes and E4M3FN row-group scales in row_interleaved_v1 layout.

Each row is K/2 code bytes (element 2j in the low nibble of byte j) followed by K/16 scale bytes.
The tensor-wide FP32 multiplier G is a separate parameter. Dequantization (one rule for every
reference and kernel): bf16_rn((fp32(e2m1) * fp32(e4m3)) * G), computed in FP32.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..layouts import row_interleaved_geometry
from ._tensor_bytes import (
    Payload,
    _exact_uint8_matrix,
    _payload_length,
    _payload_tensor,
)

FORMAT = "nvfp4_row_e4m3_g16"
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def validate_nvfp4_row_scales(scales: torch.Tensor) -> None:
    if bool(((scales & 0x80) != 0).any() | (scales == 0x7F).any()):
        raise ValueError("row-scaled NVFP4 scales must be nonnegative finite E4M3FN words")


def encode_nvfp4_rows(code_bytes: torch.Tensor, scale_words: torch.Tensor, shape: Sequence[int]) -> bytes:
    """Interleave exact code bytes [N, K/2] and E4M3FN scale words [N, K/16] row by row."""

    geometry = row_interleaved_geometry(FORMAT, shape)
    codes = _exact_uint8_matrix(code_bytes, (geometry.n, geometry.code_row_bytes), "row-scaled NVFP4 codes")
    scales = _exact_uint8_matrix(scale_words, (geometry.n, geometry.scale_row_bytes), "row-scaled NVFP4 scales")
    validate_nvfp4_row_scales(scales)
    return torch.cat((codes, scales), dim=1).contiguous().numpy().tobytes()


def decode_nvfp4_row_words(payload: Payload, shape: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact code bytes [N, K/2] and scale words [N, K/16] of a payload."""

    geometry = row_interleaved_geometry(FORMAT, shape)
    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError(
            f"row-scaled NVFP4 payload has {_payload_length(payload)} bytes, expected {geometry.payload_bytes}"
        )
    rows = _payload_tensor(payload, torch.device("cpu")).reshape(geometry.n, geometry.row_bytes)
    codes = rows[:, : geometry.code_row_bytes].clone()
    scales = rows[:, geometry.code_row_bytes :].clone()
    validate_nvfp4_row_scales(scales)
    return codes, scales


def dequantize_nvfp4_rows(
    payload: Payload,
    shape: Sequence[int],
    global_scale: float,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reconstruct the table as bf16_rn((e2m1 * e4m3) * G) in FP32 (BF16 by default)."""

    codes, scales = decode_nvfp4_row_words(payload, shape)
    table = torch.tensor(_E2M1, dtype=torch.float32)
    low = table[(codes & 0xF).long()]
    high = table[(codes >> 4).long()]
    elements = torch.stack((low, high), dim=2).reshape(codes.shape[0], -1)
    multipliers = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    g = torch.tensor(global_scale, dtype=torch.float32)
    return ((elements * multipliers) * g).to(dtype)
