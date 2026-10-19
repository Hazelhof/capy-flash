# Reader of one sequence of the compact dump (NINFER_QWEN4_TFDUMP_COMPACT=1, _TOPK=k).
import numpy as np


def read_hidden(path):
    """hidden_LXX.bin -> (positions int32 [T], bf16 bits uint16 [T, rows])."""
    raw = open(path, "rb").read()
    off, pos, chunks = 0, [], []
    while off < len(raw):
        first, cols, rows = np.frombuffer(raw, np.int32, 3, off)
        off += 12
        n = int(cols) * int(rows)
        chunks.append(np.frombuffer(raw, np.uint16, n, off).reshape(int(cols), int(rows)))
        off += 2 * n
        pos.append(np.arange(first, first + cols, dtype=np.int32))
    return np.concatenate(pos), np.concatenate(chunks)


def read_topk(path, k=20):
    rec = np.dtype([("pos", np.int32), ("ids", np.int32, k), ("lp", np.float32, k)])
    return np.fromfile(path, rec)


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)
