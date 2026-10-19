#!/bin/bash
# Artifact with the refit dense MTP part: a shadow checkpoint (symlinks to the NVIDIA checkpoint + the file of the new
# mtp.* tensors, the index pointing at it), then tools/capy_flash/convert.py (product recipe + draft shortlist).
#   usage (from the repository root): make_artifact.sh <checkpoint NVIDIA> <mtp_refit.safetensors> <shadow dir> <out dir>
set -eu
SRC=$1; REFIT=$2; SH=$3; OUT=$4; PY=${PY:-python3}
rm -rf $SH; mkdir -p $SH
for f in $SRC/*; do b=$(basename $f); [ $b = model.safetensors.index.json ] || ln -s $(readlink -f $f) $SH/$b; done
cp $REFIT $SH/mtp-refit.safetensors
$PY - $SRC $SH <<'PY'
import json, sys
from safetensors import safe_open
src, sh = sys.argv[1:]
idx = json.load(open(f"{src}/model.safetensors.index.json"))
with safe_open(f"{sh}/mtp-refit.safetensors", "pt") as f:
    names = list(f.keys())
missing = [n for n in names if n not in idx["weight_map"]]
assert not missing, missing
for n in names:
    idx["weight_map"][n] = "mtp-refit.safetensors"
json.dump(idx, open(f"{sh}/model.safetensors.index.json", "w"), indent=1)
print("replaced", len(names), "tensors")
PY
$PY tools/capy_flash/convert.py --model $SH --out $OUT
