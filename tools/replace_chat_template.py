#!/usr/bin/env python3
"""Replace the chat template stored in a NInfer v3 artifact, keeping every weight byte in place.

Run: python3 tools/replace_chat_template.py ENTRY.ninfer TEMPLATE.jinja OUT_DIR

The new template is appended after the last payload byte (in the last file) and the directory
entry `resource/text/chat_template.jinja` points to it; the directory keeps its length (it is
padded with spaces), so every other object keeps its file and offset. OUT_DIR receives the
complete artifact: the entry file and the last part are copied and edited, the parts in between
are hard links to the input (read-only for this tool). The input artifact is never modified.

Typical use: give an artifact converted with Qwen's upstream template the maintained
tools/chat_templates/qwen3_8.jinja (developer role, system turns after the first message).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import struct

HEADER = struct.Struct("<8sQ16s")
ENTRY_MAGIC = b"NINFER\0\3"
PART_MAGIC = b"NINPRT\0\3"
PAYLOAD_ALIGNMENT = 4096
LIMIT = 32_000_000_000
TEMPLATE_ID = "resource/text/chat_template.jinja"


def align(value: int, amount: int) -> int:
    return (value + amount - 1) // amount * amount


def read_directory(entry: Path):
    with entry.open("rb") as f:
        magic, json_bytes, identity = HEADER.unpack(f.read(HEADER.size))
        if magic != ENTRY_MAGIC:
            raise SystemExit(f"{entry}: not a NInfer v3 entry file")
        text = f.read(json_bytes)
    return json.loads(text), json_bytes, identity


def replace(entry: Path, template_path: Path, out_dir: Path) -> None:
    directory, json_bytes, identity = read_directory(entry)
    template = template_path.read_bytes()
    objects = directory["objects"]
    found = [o for o in objects if o["id"] == TEMPLATE_ID]
    if len(found) != 1 or found[0].get("kind") != "resource":
        raise SystemExit(f"{entry}: no single {TEMPLATE_ID} resource")
    target = found[0]
    files = directory["files"]
    payload = sum(f["payload_bytes"] for f in files)
    if any(o["offset"] + o["bytes"] > payload for o in objects):
        raise SystemExit(f"{entry}: directory payload is inconsistent")

    # Append after the last payload byte: every existing object keeps its offset.
    offset = align(payload, 256)
    grow = offset + len(template) - payload
    last = len(files) - 1
    start = align(HEADER.size + json_bytes, PAYLOAD_ALIGNMENT) if last == 0 else PAYLOAD_ALIGNMENT
    if start + files[last]["payload_bytes"] + grow > LIMIT:
        raise SystemExit("the last file would exceed the 32 GB file limit")
    target.update(offset=offset, bytes=len(template))
    objects.remove(target)
    objects.append(target)
    files[last]["payload_bytes"] += grow
    data = json.dumps(directory, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    if len(data) > json_bytes:
        raise SystemExit("the edited directory does not fit the original directory length")
    data += b" " * (json_bytes - len(data))

    names = [entry.name] + [f["path"] for f in files[1:]]
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [out_dir / n for n in names]
    if any(p.exists() for p in outputs):
        raise SystemExit(f"{out_dir}: output files already exist")
    for index, (name, out) in enumerate(zip(names, outputs)):
        source = entry.parent / name
        if index not in (0, last):
            os.link(source, out)
            continue
        tmp = out.with_name(f".{out.name}.tmp")
        shutil.copyfile(source, tmp)
        with tmp.open("r+b") as f:
            if index == 0:
                f.seek(HEADER.size)
                f.write(data)
            if index == last:
                f.seek(0, os.SEEK_END)
                f.write(bytes(grow - len(template)))
                f.write(template)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, out)

    # Check what a reader checks: lengths match the directory, the template reads back.
    check, check_bytes, check_id = read_directory(outputs[0])
    if check_bytes != json_bytes or check_id != identity:
        raise SystemExit("verification failed: header changed")
    for index, (record, out) in enumerate(zip(check["files"], outputs)):
        begin = align(HEADER.size + check_bytes, PAYLOAD_ALIGNMENT) if index == 0 else PAYLOAD_ALIGNMENT
        if out.stat().st_size != begin + record["payload_bytes"]:
            raise SystemExit(f"verification failed: {out.name} length differs from the directory")
    with outputs[last].open("rb") as f:
        f.seek(-len(template), os.SEEK_END)
        if f.read() != template:
            raise SystemExit("verification failed: template bytes")
    print(f"{outputs[0]}: chat template replaced ({len(template)} bytes at payload offset {offset})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("entry", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    if args.out_dir.resolve() == args.entry.parent.resolve():
        raise SystemExit("OUT_DIR must differ from the input directory")
    replace(args.entry, args.template, args.out_dir)


if __name__ == "__main__":
    main()
