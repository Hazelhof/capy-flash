#!/bin/bash
# capy-flash: the small Qwen3.8-Flash-Next test artifact of the Engine integration tests
# (NINFER_TEST_QWEN4_ARTIFACT: ninfer_qwen4_tp2_engine_test, ninfer_qwen4_mtp_engine_test).
#
#   tools/capy_flash/make_test_artifact.sh <NVIDIA Qwen3.8-Flash-Next-NVFP4 checkpoint dir> <out dir>
#
# 1. gen_synthetic.py (sybrix, Apache-2.0): a random checkpoint with the REAL per-layer geometry
#    (hidden 2560, 4 hyper-connection streams, 512 NVFP4 experts top-10 + shared, GDN and QSA layers,
#    PLE at block 1, MTP head), only 4 Text layers (GDN GDN GDN QSA) and a tiny n-gram table; tokenizer
#    and config come from the real checkpoint (only those files are read). Seed 1234. ~11 GB, CPU only.
# 2. tools.convert --recipe qwen3_8_flash_next_nvfp4 --components text,mtp (CPU, under a minute).
# Prints the export line for the tests. Needs Python with torch, numpy and safetensors.
set -euo pipefail
[ $# -eq 2 ] || { sed -n 2,13p "$0"; exit 2; }
SRC=$1; OUT=$2; PY=${PYTHON:-python3}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"
[ -f "$OUT/fn4/model.safetensors.index.json" ] ||
  "$PY" "$ROOT/tools/capy_flash/gen_synthetic.py" --source "$SRC" --out "$OUT/fn4" --layers 4 --mtp --seed 1234
(cd "$ROOT" && "$PY" -m tools.convert --model "$OUT/fn4" --recipe qwen3_8_flash_next_nvfp4 --components text,mtp \
   --device cpu --name capy-flash-fn4 --out "$OUT/fn4.ninfer")
echo "export NINFER_TEST_QWEN4_ARTIFACT=$OUT/fn4.ninfer"
