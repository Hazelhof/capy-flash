"""Qwen3.8-Flash-Next routed-expert banks from a GGUF (GSQ-RCO), kept as ggml blocks.

The GSQ-RCO GGUF (ISTA-DASLab, `Qwen3.8-Flash-Next-GSQ-RCO-*.gguf`) quantizes every routed
expert bank with a per-layer mix of ggml block types (IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS for gate/up,
IQ4_NL or Q2_0 for down). This tool takes ONLY those banks and stores them in the engine's bank
geometry with the `gguf_blocks_v1` layout (tools/artifact/layouts.py): the blocks are copied byte
for byte, never decoded or re-encoded.

  gate|up parent [E*2I, H]: expert e = its 640 gate rows, then its 640 up rows (expert-major, the
                            row order of the NVFP4 bank; the GGUF keeps gate and up apart)
  down parent    [E*H, I]:  expert e = its 2560 rows (the GGUF order already)

Everything else (dense, PLE, MTP-B, its NVFP4 experts) comes from the current .ninfer artifact.

Subcommands (pure Python, no numerical libraries):
  inventory  tensor types of the GGUF, the per-layer bank formats and sizes, and a check that the
             GGUF expert banks match the artifact's bank objects (shapes, layers, expert count)
  extract    write the banks of selected layers (`--layers 0,1,17`) as two files per layer plus
             manifest.json (formats, shapes, row bytes, SHA-256)
  verify     round trip: rebuild the GGUF tensors gate, up and down from the extracted banks and
             compare them byte for byte with the GGUF
  assemble   the full artifact: the source artifact's directory with the 96 Text expert objects
             replaced (format gguf_*, layout gguf_blocks_v1), their activation divisors dropped
             (Uses become A16Only) and every other object copied; `--plan-only` builds and checks
             the directory and prints the sizes without writing payload. With --mtp-iq4nl (the
             default) the MTP layer's NVFP4 expert banks are re-encoded as IQ4_NL (ggml's
             quantize_iq4_nl of the exactly decoded NVFP4 values, same bytes per expert) so every
             routed-expert layer of the artifact is a ggml block bank; the MTP drafter's other
             weights are copied unchanged. Needs numpy and a C compiler for third_party/ggml-iq.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE, GgufFormat
from tools.artifact.layouts import gguf_blocks_geometry

GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32
# GGUF metadata value types (gguf.h): scalar widths, 8 = string, 9 = array.
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
# ggml type ids of tensors outside the expert banks (only for the inventory's names).
_GGML_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
               13: "Q5_K", 14: "Q6_K", 20: "IQ4_NL", 23: "IQ4_XS", 30: "BF16"}

HIDDEN = 2560
INTERMEDIATE = 640
EXPERTS = 512
LAYERS = 48
COPY_CHUNK = 64 << 20


@dataclass(frozen=True)
class GgufTensor:
    name: str
    dims: tuple[int, ...]  # ggml order: dims[0] is the contiguous (K) axis
    ggml_type: int
    path: Path
    offset: int  # absolute file offset
    nbytes: int


def _read_string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")


def _skip_value(f, vtype: int):
    if vtype in _SCALAR:
        fmt = "<" + _SCALAR[vtype]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if vtype == 8:
        return _read_string(f)
    if vtype == 9:
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if etype in _SCALAR:
            width = struct.calcsize("<" + _SCALAR[etype])
            f.seek(width * count, os.SEEK_CUR)
            return f"<array {count}>"
        for _ in range(count):
            _skip_value(f, etype)
        return f"<array {count}>"
    raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf(path: Path) -> tuple[dict, list[GgufTensor]]:
    """Header, metadata (scalars and strings) and tensor directory of one GGUF file."""
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        if version not in (2, 3):
            raise ValueError(f"{path}: GGUF version {version} is not supported")
        meta = {}
        for _ in range(n_kv):
            key = _read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            meta[key] = _skip_value(f, vtype)
        infos = []
        for _ in range(n_tensors):
            name = _read_string(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            ggml_type, rel = struct.unpack("<IQ", f.read(12))
            infos.append((name, tuple(dims), ggml_type, rel))
        alignment = int(meta.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
        data_start = (f.tell() + alignment - 1) // alignment * alignment
    tensors = []
    for name, dims, ggml_type, rel in infos:
        tensors.append(GgufTensor(name, dims, ggml_type, path, data_start + rel, _tensor_bytes(dims, ggml_type)))
    return meta, tensors


def _tensor_bytes(dims, ggml_type) -> int:
    spec = GGUF_FORMATS_BY_TYPE.get(ggml_type)
    count = 1
    for d in dims:
        count *= d
    if spec is not None:
        return count // spec.block_elements * spec.block_bytes
    # Block sizes of the dense types (ggml-common.h), for the inventory's byte totals only.
    blocks = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34), 2: (32, 18), 10: (256, 84), 11: (256, 110),
              12: (256, 144), 13: (256, 176), 14: (256, 210)}
    if ggml_type not in blocks:
        return -1
    elements, nbytes = blocks[ggml_type]
    return count // elements * nbytes


def open_gguf(paths: list[Path]) -> tuple[dict, dict[str, GgufTensor]]:
    meta, out = {}, {}
    for i, p in enumerate(paths):
        m, tensors = read_gguf(p)
        if i == 0:
            meta = m
        for t in tensors:
            if t.name in out:
                raise ValueError(f"tensor {t.name} appears in more than one GGUF file")
            out[t.name] = t
    return meta, out


def type_name(ggml_type: int) -> str:
    spec = GGUF_FORMATS_BY_TYPE.get(ggml_type)
    return spec.name.removeprefix("gguf_").upper() if spec else _GGML_NAMES.get(ggml_type, str(ggml_type))


@dataclass(frozen=True)
class LayerBanks:
    layer: int
    gate: GgufTensor
    up: GgufTensor
    down: GgufTensor

    @property
    def gu_format(self) -> GgufFormat:
        return GGUF_FORMATS_BY_TYPE[self.gate.ggml_type]

    @property
    def down_format(self) -> GgufFormat:
        return GGUF_FORMATS_BY_TYPE[self.down.ggml_type]

    def gu_row_bytes(self) -> int:
        return gguf_blocks_geometry(self.gu_format, (2 * INTERMEDIATE * EXPERTS, HIDDEN)).row_bytes

    def down_row_bytes(self) -> int:
        return gguf_blocks_geometry(self.down_format, (HIDDEN * EXPERTS, INTERMEDIATE)).row_bytes

    def expert_bytes(self) -> int:
        return 2 * INTERMEDIATE * self.gu_row_bytes() + HIDDEN * self.down_row_bytes()


def layer_banks(tensors: dict[str, GgufTensor]) -> list[LayerBanks]:
    """The 48 layers' routed-expert banks, checked against the Flash-Next geometry."""
    out = []
    for layer in range(LAYERS):
        get = lambda role: tensors[f"blk.{layer}.ffn_{role}_exps.weight"]
        gate, up, down = get("gate"), get("up"), get("down")
        if gate.dims != (HIDDEN, INTERMEDIATE, EXPERTS) or up.dims != gate.dims:
            raise ValueError(f"layer {layer}: gate/up banks are {gate.dims}/{up.dims}, not [2560,640,512]")
        if down.dims != (INTERMEDIATE, HIDDEN, EXPERTS):
            raise ValueError(f"layer {layer}: down bank is {down.dims}, not [640,2560,512]")
        if gate.ggml_type != up.ggml_type:
            raise ValueError(f"layer {layer}: gate and up have different block types")
        for t in (gate, up, down):
            if t.ggml_type not in GGUF_FORMATS_BY_TYPE:
                raise ValueError(f"{t.name}: block type {t.ggml_type} has no gguf_* format")
        lb = LayerBanks(layer, gate, up, down)
        if gate.nbytes != EXPERTS * INTERMEDIATE * lb.gu_row_bytes() or down.nbytes != EXPERTS * HIDDEN * lb.down_row_bytes():
            raise ValueError(f"layer {layer}: bank byte counts do not match the block geometry")
        out.append(lb)
    if any(re.match(rf"blk\.{LAYERS}\.", name) for name in tensors):
        raise ValueError("the GGUF has layers beyond 48 (an MTP layer?): not expected")
    return out


def _copy(src, dst, offset: int, length: int, digest=None):
    src.seek(offset)
    while length:
        chunk = src.read(min(length, COPY_CHUNK))
        if not chunk:
            raise IOError("short read")
        dst.write(chunk)
        if digest is not None:
            digest.update(chunk)
        length -= len(chunk)


def iter_gate_up(lb: LayerBanks, chunk_experts: int = 64):
    """The gate|up bank payload in engine order (expert-major, gate then up), in chunks."""
    expert_matrix = INTERMEDIATE * lb.gu_row_bytes()
    with open(lb.gate.path, "rb") as g, open(lb.up.path, "rb") as u:
        for e0 in range(0, EXPERTS, chunk_experts):
            n = min(chunk_experts, EXPERTS - e0)
            g.seek(lb.gate.offset + e0 * expert_matrix)
            u.seek(lb.up.offset + e0 * expert_matrix)
            gate, up = g.read(n * expert_matrix), u.read(n * expert_matrix)
            if len(gate) != n * expert_matrix or len(up) != n * expert_matrix:
                raise IOError("short read")
            out = bytearray(2 * n * expert_matrix)
            for i in range(n):
                out[2 * i * expert_matrix:(2 * i + 1) * expert_matrix] = gate[i * expert_matrix:(i + 1) * expert_matrix]
                out[(2 * i + 1) * expert_matrix:(2 * i + 2) * expert_matrix] = up[i * expert_matrix:(i + 1) * expert_matrix]
            yield bytes(out)


def iter_down(lb: LayerBanks):
    with open(lb.down.path, "rb") as f:
        f.seek(lb.down.offset)
        left = lb.down.nbytes
        while left:
            chunk = f.read(min(left, COPY_CHUNK))
            if not chunk:
                raise IOError("short read")
            yield chunk
            left -= len(chunk)


def _parse_layers(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    if any(not 0 <= x < LAYERS for x in out):
        raise ValueError("layers must lie in 0..47")
    return out


# ------------------------------------------------------------------ artifact side
_EXPERT_BINDING = re.compile(r"^(text|mtp)/layers/(\d+)/moe/experts/(\d+)/(gate|up|down)$")


def artifact_expert_objects(artifact) -> dict[tuple[str, int, str], str]:
    """(component, layer, 'gate_up' | 'down') -> object id of the bank parent, from the bindings."""
    out: dict[tuple[str, int, str], str] = {}
    for name, binding in artifact.directory.bindings.items():
        m = _EXPERT_BINDING.match(name)
        if not m:
            continue
        comp, layer, _expert, role = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        parts = binding.get("parts")
        if not parts or len(parts) != 1:
            raise ValueError(f"{name}: an expert binds one region of a bank")
        key = (comp, layer, "down" if role == "down" else "gate_up")
        if out.setdefault(key, parts[0]["object"]) != parts[0]["object"]:
            raise ValueError(f"{name}: experts of one layer use several bank objects")
    return out


def check_against_artifact(banks: list[LayerBanks], artifact_path: Path) -> dict:
    from tools.artifact.reader import Artifact

    with Artifact(artifact_path) as artifact:
        objects = {obj.id: obj for obj in artifact.objects}
        experts = artifact_expert_objects(artifact)
        text_layers = sorted({layer for (comp, layer, _r) in experts if comp == "text"})
        if text_layers != list(range(LAYERS)):
            raise ValueError(f"artifact Text expert layers are {text_layers}")
        nvfp4_ids = {obj.id for obj in artifact.objects if getattr(obj, "format", None) == "nvfp4"}
        bank_ids = set(experts.values())
        report = {"artifact_nvfp4_objects": len(nvfp4_ids), "artifact_expert_bank_objects": len(bank_ids),
                  "mtp_bank_objects": sorted(v for (c, _l, _r), v in experts.items() if c == "mtp")}
        if not bank_ids <= nvfp4_ids:
            raise ValueError("some expert bank is not NVFP4")
        report["non_expert_nvfp4_objects"] = sorted(nvfp4_ids - bank_ids)
        replaced_old = replaced_new = 0
        for lb in banks:
            gu = objects[experts[("text", lb.layer, "gate_up")]]
            dn = objects[experts[("text", lb.layer, "down")]]
            if tuple(gu.shape) != (EXPERTS * 2 * INTERMEDIATE, HIDDEN) or tuple(dn.shape) != (EXPERTS * HIDDEN, INTERMEDIATE):
                raise ValueError(f"layer {lb.layer}: artifact bank shapes {gu.shape} {dn.shape}")
            replaced_old += gu.bytes + dn.bytes
            replaced_new += EXPERTS * lb.expert_bytes()
        mtp_bytes = sum(objects[v].bytes for (c, _l, _r), v in experts.items() if c == "mtp")
        report.update(text_bank_bytes_nvfp4=replaced_old, text_bank_bytes_gguf=replaced_new,
                      mtp_bank_bytes_nvfp4=mtp_bytes, artifact_payload_bytes=artifact.payload_bytes)
        return report


def cmd_inventory(args) -> None:
    meta, tensors = open_gguf(args.gguf)
    print(f"name: {meta.get('general.name')}  quantized_by: {meta.get('general.quantized_by')}  "
          f"tensors: {len(tensors)}  file_type: {meta.get('general.file_type')}")
    by_role: dict[tuple[str, str], int] = {}
    for t in tensors.values():
        m = re.match(r"blk\.\d+\.(.*)", t.name)
        role = m.group(1) if m else t.name
        by_role[(role, type_name(t.ggml_type))] = by_role.get((role, type_name(t.ggml_type)), 0) + 1
    banks = layer_banks(tensors)
    expert_names = {t.name for lb in banks for t in (lb.gate, lb.up, lb.down)}
    print("roles kept from the GGUF: ffn_{gate,up,down}_exps of blk.0..47 (%d tensors); every other "
          "tensor (%d) stays from the artifact" % (len(expert_names), len(tensors) - len(expert_names)))
    if args.verbose:
        for (role, ty), n in sorted(by_role.items()):
            print(f"  {role:34s} {ty:8s} {n}")
    total = 0
    combos: dict[tuple[str, str], list[int]] = {}
    for lb in banks:
        total += EXPERTS * lb.expert_bytes()
        combos.setdefault((type_name(lb.gate.ggml_type), type_name(lb.down.ggml_type)), []).append(lb.layer)
    print("layer banks (gate/up + down: layers, bytes per expert):")
    for (gu, dn), layers in sorted(combos.items()):
        print(f"  {gu:8s} + {dn:7s} {len(layers):2d} layers {banks[layers[0]].expert_bytes():>9} B  {layers}")
    print(f"Text expert banks: {total} B = {total / 2**30:.3f} GiB, mean {total / (LAYERS * EXPERTS):.1f} B/expert")
    summary = {"text_bank_bytes_gguf": total,
               "layers": [{"layer": lb.layer, "gate_up": lb.gu_format.name, "down": lb.down_format.name,
                           "expert_bytes": lb.expert_bytes()} for lb in banks]}
    if args.artifact:
        report = check_against_artifact(banks, args.artifact)
        summary["artifact"] = report
        old, new, mtp = report["text_bank_bytes_nvfp4"], report["text_bank_bytes_gguf"], report["mtp_bank_bytes_nvfp4"]
        print(f"artifact: {report['artifact_expert_bank_objects']} expert bank objects "
              f"(MTP {report['mtp_bank_objects']}), non-expert NVFP4 objects {report['non_expert_nvfp4_objects']}")
        print(f"  Text banks NVFP4 {old / 2**30:.3f} GiB -> GGUF {new / 2**30:.3f} GiB; MTP banks (kept NVFP4) "
              f"{mtp / 2**30:.3f} GiB; host-mapped experts total {(new + mtp) / 2**30:.3f} GiB "
              f"(today {(old + mtp) / 2**30:.3f})")
        print(f"  artifact payload {report['artifact_payload_bytes'] / 2**30:.2f} GiB -> "
              f"{(report['artifact_payload_bytes'] - old + new) / 2**30:.2f} GiB")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1))


def cmd_extract(args) -> None:
    _meta, tensors = open_gguf(args.gguf)
    banks = layer_banks(tensors)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"layout": "gguf_blocks_v1", "layers": {}}
    for layer in _parse_layers(args.layers):
        lb = banks[layer]
        entry = {}
        for role, fmt, shape, chunks in (
            ("gate_up", lb.gu_format, (EXPERTS * 2 * INTERMEDIATE, HIDDEN), iter_gate_up(lb)),
            ("down", lb.down_format, (EXPERTS * HIDDEN, INTERMEDIATE), iter_down(lb)),
        ):
            geo = gguf_blocks_geometry(fmt, shape)
            digest = hashlib.sha256()
            path = out / f"layer_{layer:02d}.{role}.bin"
            written = 0
            with open(path, "wb") as f:
                for chunk in chunks:
                    f.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
            if written != geo.payload_bytes:
                raise ValueError(f"{path}: wrote {written} bytes, expected {geo.payload_bytes}")
            entry[role] = {"file": path.name, "format": fmt.name, "shape": list(shape), "row_bytes": geo.row_bytes,
                           "bytes": geo.payload_bytes, "sha256": digest.hexdigest()}
        entry["expert_bytes"] = lb.expert_bytes()
        manifest["layers"][str(layer)] = entry
        print(f"layer {layer}: {entry['gate_up']['format']} {entry['gate_up']['bytes']} B + "
              f"{entry['down']['format']} {entry['down']['bytes']} B ({lb.expert_bytes()} B/expert)")
    manifest_path.write_text(json.dumps(manifest, indent=1))


def cmd_verify(args) -> None:
    """Rebuild gate, up and down of the GGUF from the extracted banks: byte-exact or fail."""
    _meta, tensors = open_gguf(args.gguf)
    banks = layer_banks(tensors)
    out = Path(args.banks)
    manifest = json.loads((out / "manifest.json").read_text())
    failures = 0
    for key, entry in sorted(manifest["layers"].items(), key=lambda kv: int(kv[0])):
        lb = banks[int(key)]
        expert_matrix = INTERMEDIATE * lb.gu_row_bytes()
        checks = []
        gu_path, dn_path = out / entry["gate_up"]["file"], out / entry["down"]["file"]
        # Rebuilt GGUF tensors (gate = even expert halves, up = odd) vs the GGUF bytes.
        for role, tensor, parity in (("gate", lb.gate, 0), ("up", lb.up, 1)):
            mine, theirs = hashlib.sha256(), hashlib.sha256()
            with open(gu_path, "rb") as f:
                for e in range(EXPERTS):
                    f.seek((2 * e + parity) * expert_matrix)
                    mine.update(f.read(expert_matrix))
            with open(tensor.path, "rb") as g:
                g.seek(tensor.offset)
                left = tensor.nbytes
                while left:
                    chunk = g.read(min(left, COPY_CHUNK))
                    theirs.update(chunk)
                    left -= len(chunk)
            checks.append((role, mine.hexdigest() == theirs.hexdigest()))
        mine, theirs = hashlib.sha256(), hashlib.sha256()
        with open(dn_path, "rb") as f:
            for chunk in iter(lambda: f.read(COPY_CHUNK), b""):
                mine.update(chunk)
        for chunk in iter_down(lb):
            theirs.update(chunk)
        checks.append(("down", mine.hexdigest() == theirs.hexdigest()))
        # Files against the manifest digests.
        for role in ("gate_up", "down"):
            d = hashlib.sha256()
            with open(out / entry[role]["file"], "rb") as f:
                for chunk in iter(lambda: f.read(COPY_CHUNK), b""):
                    d.update(chunk)
            checks.append((f"{role}.sha256", d.hexdigest() == entry[role]["sha256"]))
        ok = all(v for _k, v in checks)
        failures += not ok
        print(f"layer {key}: " + " ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in checks))
    if failures:
        sys.exit(f"{failures} layer(s) failed the round trip")
    print("round trip byte-exact on every extracted layer")


# ------------------------------------------------------------------ MTP banks: NVFP4 -> IQ4_NL
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def ggml_iq_library():
    """ctypes handle of third_party/ggml-iq built as a shared library (cached per process)."""
    import ctypes
    import subprocess
    import tempfile

    src = Path(__file__).resolve().parents[2] / "third_party" / "ggml-iq" / "ggml-iq.c"
    out = Path(tempfile.gettempdir()) / f"ggml_iq_{os.getuid()}_{int(src.stat().st_mtime)}.so"
    if not out.exists():
        subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-ffp-contract=off", "-fno-fast-math",
                        "-I", str(src.parent), str(src), "-o", str(out), "-lm"], check=True)
    lib = ctypes.CDLL(str(out))
    lib.quantize_iq4_nl.restype = ctypes.c_size_t
    lib.quantize_iq4_nl.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    lib.dequantize_row_iq4_nl.restype = None
    lib.dequantize_row_iq4_nl.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    return lib


def nvfp4_rows(artifact, obj, row0: int, rows: int):
    """Exact float32 values of rows [row0, row0 + rows) of an NVFP4 block_scale_k16_m128x4_v1 object
    (rows and row0 multiples of 128): e2m1 code * e4m3 scale / divisor of the row's source matrix."""
    import numpy as np
    from tools.artifact.formats import decode_e4m3fn_word

    n, k = obj.shape
    code_plane = n * k // 2
    scale_off = (code_plane + 255) // 256 * 256
    div_off = scale_off + n * k // 16
    codes = np.frombuffer(artifact.read_range(obj.offset + row0 * k // 2, rows * k // 2), np.uint8).reshape(rows, k // 2)
    tiles, ktiles = rows // 128, k // 64
    raw = np.frombuffer(artifact.read_range(obj.offset + scale_off + row0 // 128 * ktiles * 512, tiles * ktiles * 512), np.uint8)
    # [tile, scale_tile, row%32, row//32 (of the tile), lane] -> [tile, row//32, row%32, scale_tile, lane]
    sc = raw.reshape(tiles, ktiles, 32, 4, 4).transpose(0, 3, 2, 1, 4).reshape(rows, k // 16)
    per = n // obj.divisors
    divs = np.frombuffer(artifact.read_range(obj.offset + div_off, 4 * obj.divisors), "<f4")
    row_div = divs[(row0 + np.arange(rows)) // per].astype(np.float32)
    e2m1 = np.array(_E2M1, np.float32)
    e4m3 = np.array([decode_e4m3fn_word(i) for i in range(256)], np.float32)
    vals = np.empty((rows, k), np.float32)
    vals[:, 0::2] = e2m1[codes & 15]
    vals[:, 1::2] = e2m1[codes >> 4]
    vals *= np.repeat(e4m3[sc], 16, axis=1)
    vals /= row_div[:, None]
    return vals


def iter_mtp_iq4nl(artifact, obj, rows_per_expert: int, lib, report: list, threads: int = 8):
    """The IQ4_NL bank of an NVFP4 expert bank, expert by expert, quantized on `threads` threads."""
    import ctypes
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    n, k = obj.shape
    row_bytes = k // 32 * 18

    def one(e):
        vals = np.ascontiguousarray(nvfp4_rows(artifact, obj, e * rows_per_expert, rows_per_expert))
        out = np.empty(rows_per_expert * row_bytes, np.uint8)
        lib.quantize_iq4_nl(vals.ctypes.data, out.ctypes.data, rows_per_expert, k, None)
        if e % 64 == 0:  # requantization error of a sample row block
            back = np.empty(16 * k, np.float32)
            lib.dequantize_row_iq4_nl(out.ctypes.data, back.ctypes.data, 16 * k)
            ref = vals[:16].reshape(-1)
            report.append(float(np.sqrt(((back - ref) ** 2).sum() / max((ref ** 2).sum(), 1e-30))))
        return out.tobytes()

    experts = n // rows_per_expert
    with ThreadPoolExecutor(threads) as pool:
        for e0 in range(0, experts, threads):
            for chunk in pool.map(one, range(e0, min(experts, e0 + threads))):
                yield chunk


def cmd_assemble(args) -> None:
    from tools.artifact.reader import Artifact
    from tools.artifact.schema import ResourceObject, ResourceSpec, TensorObject, TensorSpec
    from tools.artifact.writer import ArtifactWriter, layout_directory, DEFAULT_MAX_FILE_BYTES
    from tools.artifact.schema import plan_objects

    _meta, tensors = open_gguf(args.gguf)
    banks = layer_banks(tensors)
    with Artifact(args.artifact) as src:
        experts = artifact_expert_objects(src)
        replace: dict[str, tuple[LayerBanks, str]] = {}
        for lb in banks:
            replace[experts[("text", lb.layer, "gate_up")]] = (lb, "gate_up")
            replace[experts[("text", lb.layer, "down")]] = (lb, "down")
        mtp_replace: dict[str, int] = {}
        if args.mtp_iq4nl:
            mtp_replace[experts[("mtp", 0, "gate_up")]] = 2 * INTERMEDIATE
            mtp_replace[experts[("mtp", 0, "down")]] = HIDDEN
        # The Text experts' Uses lose their NVFP4 activation divisors (A16Only); the auxiliary
        # objects only they reference are dropped.
        uses, dropped = [], set()
        kept_aux = set()
        for use in src.directory.uses:
            m = _EXPERT_BINDING.match(use["parameter"])
            if m and m.group(1) == "text":
                for aux in use.get("auxiliaries", {}).values():
                    dropped.add(aux["object"])
                uses.append({"parameter": use["parameter"], "input": use["input"], "activation_policy": "A16Only"})
            else:
                for aux in use.get("auxiliaries", {}).values():
                    kept_aux.add(aux["object"])
                uses.append(use)
        dropped -= kept_aux
        bound = set()
        for b in src.directory.bindings.values():
            if "object" in b:
                bound.add(b["object"])
            for p in b.get("parts", ()):
                bound.add(p["object"])
        if dropped & bound:
            raise ValueError("an activation-divisor object is also bound as a parameter")
        specs, sources = [], []
        for obj in src.objects:
            if obj.id in dropped:
                continue
            if obj.id in mtp_replace:
                specs.append(TensorSpec(obj.id, tuple(obj.shape), "gguf_iq4_nl", "gguf_blocks_v1"))
                sources.append(("mtp", obj, mtp_replace[obj.id]))
            elif obj.id in replace:
                lb, role = replace[obj.id]
                fmt = lb.gu_format if role == "gate_up" else lb.down_format
                specs.append(TensorSpec(obj.id, tuple(obj.shape), fmt.name, "gguf_blocks_v1"))
                sources.append(("gguf", lb, role))
            elif isinstance(obj, TensorObject):
                specs.append(TensorSpec(obj.id, tuple(obj.shape), obj.format, obj.layout, obj.divisors))
                sources.append(("copy", obj, None))
            else:
                specs.append(ResourceSpec(obj.id, obj.bytes, obj.encoding))
                sources.append(("copy", obj, None))
        provenance = dict(src.directory.provenance)
        provenance["experts"] = {"converter": "tools.convert.gguf_experts", "gguf": [str(p) for p in args.gguf],
                                 "layout": "gguf_blocks_v1", "replaced_objects": len(replace),
                                 "mtp_experts": "nvfp4 -> gguf_iq4_nl (ggml quantize_iq4_nl of the decoded values)"
                                 if mtp_replace else "unchanged"}
        metadata = dict(src.directory.metadata)
        if args.name:
            metadata["name"] = args.name
        objects = plan_objects(specs)
        payload = objects[-1].offset + objects[-1].bytes
        print(f"objects {len(src.objects)} -> {len(objects)} (dropped {len(dropped)} activation divisors, "
              f"replaced {len(replace)} banks); payload {src.payload_bytes / 2**30:.3f} -> {payload / 2**30:.3f} GiB")
        if args.plan_only:
            description = {"components": src.directory.components, "objects": [o.to_json() for o in objects],
                           "bindings": src.directory.bindings, "uses": uses, "metadata": metadata,
                           "provenance": provenance}
            directory, encoded, _start = layout_directory(Path(args.out).name, description, payload,
                                                          max_file_bytes=DEFAULT_MAX_FILE_BYTES)
            print(f"plan only: directory {len(encoded)} B, {len(directory.files)} files; nothing written")
            return
        lib = ggml_iq_library() if mtp_replace else None
        report: list[float] = []
        with ArtifactWriter(args.out, specs, components=src.directory.components, bindings=src.directory.bindings,
                            uses=uses, metadata=metadata, provenance=provenance) as w:
            for i, (spec, (kind, item, role)) in enumerate(zip(specs, sources)):
                if kind == "copy":
                    w.write_object(spec.id, src.iter_object(item.id))
                elif kind == "mtp":
                    w.write_object(spec.id, iter_mtp_iq4nl(src, item, role, lib, report))
                    print(f"MTP {spec.id}: IQ4_NL requantization rel. L2 of sampled rows "
                          f"{min(report):.4f}..{max(report):.4f}", flush=True)
                else:
                    w.write_object(spec.id, iter_gate_up(item) if role == "gate_up" else iter_down(item))
                if i % 2000 == 0:
                    print(f"{i}/{len(specs)} objects", flush=True)
            w.finish()
        print(f"wrote {args.out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inventory")
    p.add_argument("--gguf", type=Path, nargs="+", required=True)
    p.add_argument("--artifact", type=Path)
    p.add_argument("--json")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_inventory)
    p = sub.add_parser("extract")
    p.add_argument("--gguf", type=Path, nargs="+", required=True)
    p.add_argument("--layers", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("verify")
    p.add_argument("--gguf", type=Path, nargs="+", required=True)
    p.add_argument("--banks", required=True)
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("assemble")
    p.add_argument("--gguf", type=Path, nargs="+", required=True)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--mtp-iq4nl", action=argparse.BooleanOptionalAction, default=True,
                   help="re-encode the MTP expert banks NVFP4 -> IQ4_NL (default on)")
    p.set_defaults(fn=cmd_assemble)
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
