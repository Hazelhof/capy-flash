"""Lossless decoding of CSF-compressed NVFP4 block scales (ModelOpt ``weight_scale_encoding: "csf"``).

Codec ``byte-window4-fixed-stream-u24-exceptions/1``. One ``[rows, cols]`` E4M3FN scale matrix
(``cols = K / 16``) is stored as two tensors:

- ``<name>.nvfp4_csf_fixed``: U8 ``[rows / 16, 16 * (1 + cols / 2)]``. Each row of this tensor is a
  slab of 16 scale rows: 16 base bytes (one per scale row, 0..240), then the 16 rows' 4-bit codes
  in row order, two per byte, low nibble first. A scale byte is ``base + code``.
- ``<name>.nvfp4_csf_exceptions``: U32 words in increasing position order; the low 24 bits are the
  flat position ``row * cols + col`` and the high 8 bits are the scale byte stored there instead.

The decoded bytes are the checkpoint's uncompressed E4M3FN scales, so NVFP4 import is unchanged.
"""

from __future__ import annotations

import torch

FIXED_SUFFIX = ".nvfp4_csf_fixed"
EXCEPTIONS_SUFFIX = ".nvfp4_csf_exceptions"
SLAB_ROWS = 16
MAX_BASE = 240


def csf_shape(fixed_shape: tuple[int, ...]) -> tuple[int, int]:
    """Scale matrix ``[rows, cols]`` described by a fixed-stream shape."""
    if len(fixed_shape) != 2 or fixed_shape[1] % SLAB_ROWS or fixed_shape[1] // SLAB_ROWS < 2:
        raise ValueError(f"CSF fixed stream shape {fixed_shape} is invalid")
    return fixed_shape[0] * SLAB_ROWS, 2 * (fixed_shape[1] // SLAB_ROWS - 1)


def exception_words(words: torch.Tensor, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Validated ``(positions, bytes)`` of an exception stream, both int64."""
    if words.dtype not in (torch.uint32, torch.int32) or words.dim() != 1:
        raise ValueError(f"CSF exceptions must be one U32 vector, got {words.dtype}{tuple(words.shape)}")
    wide = words.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    positions, values = wide & 0xFFFFFF, wide >> 24
    if positions.numel():
        if int(positions[-1]) >= rows * cols:
            raise ValueError("CSF exception position exceeds the scale matrix")
        if positions.numel() > 1 and not bool((positions[1:] > positions[:-1]).all()):
            raise ValueError("CSF exception positions must be strictly increasing")
    return positions, values


def decode_csf_slabs(
    fixed: torch.Tensor,
    positions: torch.Tensor,
    values: torch.Tensor,
    *,
    first_slab: int,
    cols: int,
) -> torch.Tensor:
    """Scale bytes ``[16 * slabs, cols]`` (U8) of consecutive slabs starting at ``first_slab``.

    ``fixed`` holds only those slabs; ``positions``/``values`` are the whole matrix's exceptions.
    """
    width = SLAB_ROWS * (1 + cols // 2)
    if fixed.dtype != torch.uint8 or cols % 2 or fixed.numel() % width:
        raise ValueError("CSF fixed slabs do not match the scale width")
    slabs = fixed.reshape(-1, width)
    bases = slabs[:, :SLAB_ROWS].to(torch.int32)
    if bool((bases > MAX_BASE).any()):
        raise ValueError(f"CSF row base exceeds {MAX_BASE}")
    packed = slabs[:, SLAB_ROWS:].reshape(-1, SLAB_ROWS, cols // 2).to(torch.int32)
    codes = torch.empty(packed.shape[0], SLAB_ROWS, cols, dtype=torch.int32)
    codes[..., 0::2] = packed & 0xF
    codes[..., 1::2] = packed >> 4
    out = (codes + bases[..., None]).reshape(-1, cols)
    begin = first_slab * SLAB_ROWS * cols
    end = begin + out.numel()
    low = int(torch.searchsorted(positions, begin))
    high = int(torch.searchsorted(positions, end))
    if low < high:
        out.view(-1)[positions[low:high] - begin] = values[low:high].to(torch.int32)
    return out.to(torch.uint8)


def decode_csf(fixed: torch.Tensor, exceptions: torch.Tensor) -> torch.Tensor:
    """The complete ``[rows, cols]`` U8 scale matrix of one fixed/exceptions pair."""
    rows, cols = csf_shape(tuple(fixed.shape))
    positions, values = exception_words(exceptions.reshape(-1), rows, cols)
    return decode_csf_slabs(fixed, positions, values, first_slab=0, cols=cols)
