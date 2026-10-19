"""Exercise the production Python writer through the C++ reader and materializer."""

from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.schema import TensorSpec
from tools.artifact.writer import ArtifactWriter

# Encoded fixtures written through the Python codecs (tools.artifact.tensor_output); the C++ side
# checks every word at the offsets of the layout definition (test_materialization.cpp).
MXFP8_SHAPE = (48, 64)
NVFP4_ROW_SHAPE = (6, 32)


def interop_code(row: int, index: int) -> int:
    return (row * 7 + index * 3) % 0x7E


def interop_scale(row: int, group: int) -> int:
    return (row * 5 + group * 11 + 100) % 0x7E


def _words(rows: int, columns: int, word):
    import torch

    return torch.tensor([[word(r, c) for c in range(columns)] for r in range(rows)], dtype=torch.uint8)


def _encoded_specs() -> list[TensorSpec]:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("writer interop: torch is unavailable, encoded MXFP8/NVFP4-row objects not exercised")
        return []
    return [
        TensorSpec("mxfp8", MXFP8_SHAPE, "mxfp8_e4m3fn_g32_ue8m0", "block_scale_k32_v1"),
        TensorSpec("nvfp4_rows", NVFP4_ROW_SHAPE, "nvfp4_row_e4m3_g16", "row_interleaved_v1"),
    ]


def _write_encoded(writer: ArtifactWriter) -> None:
    from tools.artifact.tensor_output import TensorOutput

    n, k = MXFP8_SHAPE
    codes, scales = _words(n, k, interop_code), _words(n, k // 32, interop_scale)
    output = TensorOutput(writer, "mxfp8")
    for begin in range(0, n, 20):  # uneven row chunks
        output.write_codes(begin, codes[begin : begin + 20], scales[begin : begin + 20])
    n, k = NVFP4_ROW_SHAPE
    codes, scales = _words(n, k // 2, interop_code), _words(n, k // 16, interop_scale)
    output = TensorOutput(writer, "nvfp4_rows")
    output.write_codes(0, codes[:4], scales[:4])
    output.write_codes(4, codes[4:], scales[4:])


def main() -> int:
    executable = sys.argv[1]
    data = bytes((index * 37 + 11) % 251 for index in range(130 * 130 * 2))
    encoded = _encoded_specs()
    with tempfile.TemporaryDirectory(prefix="ninfer-writer-interop-") as temporary:
        for label, limit in (("single", 1_000_000), ("sharded", 12288)):
            path = Path(temporary) / f"{label}.ninfer"
            with ArtifactWriter(
                path,
                [TensorSpec("matrix", (130, 130), "bf16", "contiguous_le_v1"), *encoded],
                components={"text": {"config": {}}},
                bindings={
                    "whole": {"object": "matrix"},
                    "reordered_rows": {
                        "parts": [
                            {"object": "matrix", "range": [16770, 16900]},
                            {"object": "matrix", "range": [130, 260]},
                        ]
                    },
                    **{spec.id: {"object": spec.id} for spec in encoded},
                },
                max_file_bytes=limit,
            ) as writer:
                writer.write_object("matrix", data)
                if encoded:
                    _write_encoded(writer)
            result = subprocess.run([executable, "--writer-fixture", str(path)])
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
