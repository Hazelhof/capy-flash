#!/usr/bin/env bash
# Qwen3.8-Flash-Next launcher for the release tarball: serve-flash-next.sh <artifact.ninfer> [ninfer-serve flags...]
# The capy-flash serving profile, `ninfer-serve --profile flash-next` (two 16 GB sm_120 boards, no P2P, routed experts that do not fit in VRAM on the
# CPU): INT8 KV with the elastic KV / expert-slot budget, 262,144-token context, MTP with 3 draft tokens, one
# request at a time. Needs about 70 GiB of free RAM (checked at startup). Environment overrides:
#   NINFER_HOST (127.0.0.1)  NINFER_PORT (8080)  NINFER_DEVICES (0,1)  NINFER_MAX_CONTEXT (262144)
#   NINFER_MODEL_ID (capy-flash: the `model` name requests must use)
# Extra flags are appended and win over the profile. Run it under a supervisor that restarts on a
# non-zero exit: ninfer-serve exits with status 2 once the Engine fails (docs/serving.md).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 1 ] || { echo "usage: $0 <artifact.ninfer> [ninfer-serve flags...]" >&2; exit 2; }
artifact=$1; shift
context=${NINFER_MAX_CONTEXT:-262144}
export LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$here/bin/ninfer-serve" "$artifact" --profile flash-next \
  --host "${NINFER_HOST:-127.0.0.1}" --port "${NINFER_PORT:-8080}" \
  --devices "${NINFER_DEVICES:-0,1}" --max-context "$context" --model-id "${NINFER_MODEL_ID:-capy-flash}" "$@"
