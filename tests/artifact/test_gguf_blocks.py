from tools.artifact.formats import GGUF_FORMATS, GGUF_FORMATS_BY_TYPE, get_format
from tools.artifact.layouts import encoded_size, get_layout, gguf_blocks_geometry

import pytest


def test_gguf_formats_are_ggml_block_types():
    # (ggml type id, elements per block, bytes per block) of ggml-common.h.
    expected = {
        "gguf_iq3_xxs": (18, 256, 98),
        "gguf_iq4_nl": (20, 32, 18),
        "gguf_iq3_s": (21, 256, 110),
        "gguf_iq2_s": (22, 256, 82),
        "gguf_iq4_xs": (23, 256, 136),
        "gguf_q2_0": (42, 64, 18),
    }
    assert set(GGUF_FORMATS) == set(expected)
    for name, (ggml_type, elements, nbytes) in expected.items():
        spec = get_format(name)
        assert (spec.ggml_type, spec.block_elements, spec.block_bytes) == (ggml_type, elements, nbytes)
        assert GGUF_FORMATS_BY_TYPE[ggml_type] is spec


def test_gguf_blocks_v1_sizes_flash_next_banks():
    assert get_layout("gguf_blocks_v1").alignment == 256
    # Per expert: gate|up 1280 rows of 2560, down 2560 rows of 640.
    per_expert = {
        "gguf_iq2_s": 1280 * 10 * 82,
        "gguf_iq3_xxs": 1280 * 10 * 98,
        "gguf_iq3_s": 1280 * 10 * 110,
        "gguf_iq4_xs": 1280 * 10 * 136,
    }
    for name, nbytes in per_expert.items():
        assert encoded_size("gguf_blocks_v1", name, (512 * 1280, 2560)) == 512 * nbytes
    assert encoded_size("gguf_blocks_v1", "gguf_iq4_nl", (512 * 2560, 640)) == 512 * 2560 * 20 * 18
    assert encoded_size("gguf_blocks_v1", "gguf_q2_0", (512 * 2560, 640)) == 512 * 2560 * 10 * 18
    g = gguf_blocks_geometry("gguf_iq3_s", (4, 2560))
    assert (g.blocks_per_row, g.row_bytes, g.payload_bytes) == (10, 1100, 4400)


def test_gguf_blocks_v1_rejects_partial_blocks_and_other_formats():
    with pytest.raises(ValueError):
        gguf_blocks_geometry("gguf_iq3_s", (1280, 640))  # 640 is not a whole number of superblocks
    with pytest.raises(ValueError):
        encoded_size("gguf_blocks_v1", "nvfp4", (128, 64))
    with pytest.raises(ValueError):
        encoded_size("block_scale_k16_m128x4_v1", "gguf_iq4_nl", (128, 64))
