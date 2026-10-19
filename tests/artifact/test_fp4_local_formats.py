"""MXFP8 (block_scale_k32_v1) and row-scaled NVFP4 (row_interleaved_v1): verbatim word codecs.

The oracle decodes every element from the format definition, one Python scalar at a time; the
codecs are only asked to place and return the exact words.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.artifact.codecs.mxfp8 import decode_mxfp8_words, dequantize_mxfp8, encode_mxfp8
from tools.artifact.codecs.nvfp4_row import decode_nvfp4_row_words, dequantize_nvfp4_rows, encode_nvfp4_rows
from tools.artifact.formats import decode_e2m1_word, decode_e4m3fn_word
from tools.artifact.layouts import block_scale_k32_geometry, encoded_size, row_interleaved_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorSpec
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter

MXFP8 = "mxfp8_e4m3fn_g32_ue8m0"
NVFP4_ROW = "nvfp4_row_e4m3_g16"


def _words(rows: int, columns: int, seed: int, *, high: int = 256) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.integers(0, high, size=(rows, columns), dtype=np.uint8))


def _finite_e4m3(words: torch.Tensor) -> torch.Tensor:
    return torch.where((words & 0x7F) == 0x7F, words & 0x80, words)


def test_mxfp8_geometry_matches_the_layout_definition():
    g = block_scale_k32_geometry(MXFP8, (48, 2560))
    assert (g.code_plane_bytes, g.scale_plane_offset, g.scale_plane_bytes) == (122880, 122880, 3840)
    assert g.payload_bytes == 126720 == encoded_size("block_scale_k32_v1", MXFP8, (48, 2560))
    odd = block_scale_k32_geometry(MXFP8, (3, 32))
    assert (odd.code_plane_bytes, odd.scale_plane_offset, odd.payload_bytes) == (96, 256, 259)
    with pytest.raises(ValueError, match="divisible by 32"):
        block_scale_k32_geometry(MXFP8, (4, 48))
    with pytest.raises(ValueError, match="does not accept"):
        encoded_size("row_scale_v1", MXFP8, (4, 64))


def test_mxfp8_words_are_copied_verbatim_and_dequantize_exactly():
    n, k = 5, 96
    codes = _finite_e4m3(_words(n, k, 1))
    # Exponents up to 246 keep 448 * 2^(s-127) finite in FP32, so every value is exact there.
    scales = _words(n, k // 32, 2, high=247)
    scales[0, 0], scales[1, 1] = 0, 246
    payload = encode_mxfp8(codes, scales, (n, k))
    assert payload[: n * k] == codes.numpy().tobytes()
    assert payload[n * k : 512] == bytes(512 - n * k)
    assert payload[512:] == scales.numpy().tobytes()
    decoded_codes, decoded_scales = decode_mxfp8_words(payload, (n, k))
    assert torch.equal(decoded_codes, codes) and torch.equal(decoded_scales, scales)
    values = dequantize_mxfp8(payload, (n, k), torch.float64)
    for row in range(n):
        for column in range(k):
            expected = decode_e4m3fn_word(int(codes[row, column])) * 2.0 ** (int(scales[row, column // 32]) - 127)
            assert values[row, column].item() == expected, (row, column)


def test_mxfp8_rejects_nan_words():
    codes = torch.zeros((1, 32), dtype=torch.uint8)
    scales = torch.full((1, 1), 127, dtype=torch.uint8)
    bad = codes.clone()
    bad[0, 3] = 0xFF
    with pytest.raises(ValueError, match="finite E4M3FN"):
        encode_mxfp8(bad, scales, (1, 32))
    with pytest.raises(ValueError, match="UE8M0"):
        encode_mxfp8(codes, torch.full((1, 1), 255, dtype=torch.uint8), (1, 32))


def test_row_interleaved_geometry_matches_the_layout_definition():
    g = row_interleaved_geometry(NVFP4_ROW, (2_500_012, 160))
    assert (g.code_row_bytes, g.scale_row_bytes, g.row_bytes) == (80, 10, 90)
    assert g.payload_bytes == 2_500_012 * 90 == encoded_size("row_interleaved_v1", NVFP4_ROW, (2_500_012, 160))
    with pytest.raises(ValueError, match="divisible by 16"):
        row_interleaved_geometry(NVFP4_ROW, (2, 40))


def test_nvfp4_rows_interleave_verbatim_and_follow_the_rounding_rule():
    n, k = 4, 160
    codes = _words(n, k // 2, 3)
    scales = _words(n, k // 16, 4, high=0x7F)
    scales[0, 0] = 0
    g = 3.3242362e-05
    payload = encode_nvfp4_rows(codes, scales, (n, k))
    for row in range(n):
        line = payload[row * 90 : (row + 1) * 90]
        assert line[:80] == codes[row].numpy().tobytes() and line[80:] == scales[row].numpy().tobytes()
    decoded_codes, decoded_scales = decode_nvfp4_row_words(payload, (n, k))
    assert torch.equal(decoded_codes, codes) and torch.equal(decoded_scales, scales)
    values = dequantize_nvfp4_rows(payload, (n, k), g)
    assert values.dtype == torch.bfloat16
    g32 = np.float32(g)
    for row in range(n):
        for column in range(k):
            byte = int(codes[row, column // 2])
            e2m1 = decode_e2m1_word(byte & 0xF if column % 2 == 0 else byte >> 4)
            e4m3 = decode_e4m3fn_word(int(scales[row, column // 16]))
            product = (np.float32(e2m1) * np.float32(e4m3)) * g32
            expected = torch.tensor(float(product), dtype=torch.float32).to(torch.bfloat16)
            assert values[row, column].view(torch.int16) == expected.view(torch.int16), (row, column)


def test_nvfp4_rows_reject_negative_or_nan_scales():
    codes = torch.zeros((1, 8), dtype=torch.uint8)
    for word in (0x80, 0x7F, 0xFF):
        with pytest.raises(ValueError, match="nonnegative finite"):
            encode_nvfp4_rows(codes, torch.full((1, 1), word, dtype=torch.uint8), (1, 16))


def test_tensor_output_writes_row_chunks_of_both_layouts(tmp_path):
    mx_shape, row_shape = (6, 64), (7, 32)
    mx_codes, mx_scales = _finite_e4m3(_words(*mx_shape, 5)), _words(6, 2, 6, high=255)
    row_codes, row_scales = _words(7, 16, 7), _words(7, 2, 8, high=0x7F)
    specs = [
        TensorSpec("mx", mx_shape, MXFP8, "block_scale_k32_v1"),
        TensorSpec("rows", row_shape, NVFP4_ROW, "row_interleaved_v1"),
    ]
    path = tmp_path / "formats.ninfer"
    bindings = {"mx": {"object": "mx"}, "rows": {"object": "rows"}}
    with ArtifactWriter(path, specs, components={"text": {"config": {}}}, bindings=bindings) as writer:
        mx, rows = TensorOutput(writer, "mx"), TensorOutput(writer, "rows")
        for begin, end in ((4, 6), (0, 1), (1, 4)):
            mx.write_codes(begin, mx_codes[begin:end], mx_scales[begin:end])
        for begin, end in ((0, 3), (3, 7)):
            rows.write_codes(begin, row_codes[begin:end], row_scales[begin:end])
    with Artifact(path) as artifact:
        assert artifact.read_object("mx") == encode_mxfp8(mx_codes, mx_scales, mx_shape)
        assert artifact.read_object("rows") == encode_nvfp4_rows(row_codes, row_scales, row_shape)
