#!/bin/bash
# On-policy data for the MTP draft. For each TAG: sampled generation (T 1, top-k 20, top-p 0.95 = generation_config)
# at TP2 with MTP 3 and int8 KV (as served), then a compact teacher-forced dump on one GPU: the wide residual after
# layer 47 in bf16 + the model's top-20 log-probs, over prompt + answer.
#   env: DRV=<build>/tests/ninfer_qwen4_tf_driver  ART=<artifact .ninfer>  W=<work dir with PFX_TAG.ids>  PY=<python>
#   usage: gen_dump.sh <prefix train|held> off [on]
set -u
: "${DRV:?}" "${ART:?}" "${W:?}"; PY=${PY:-python3}
PFX=$1; shift
cd $W
for TAG in "$@"; do
  MAXNEW=1024; [ $TAG = on ] && MAXNEW=2048
  echo "$(date +%T) gen-${PFX}-$TAG START" >> runs.log
  env CUDA_VISIBLE_DEVICES=0,1 NINFER_DRIVER_STOPS=1 $DRV generate $ART ${PFX}_$TAG.ids --prompts --max-new $MAXNEW \
      --mtp 3 --kv int8 --tp 2 --elastic --context 4096 --temperature 1.0 --top-k 20 --top-p 0.95 --seed 194 \
      --out gen_${PFX}_$TAG.jsonl > gen_${PFX}_$TAG.log 2>&1
  echo "$(date +%T) gen-${PFX}-$TAG rc=$? $(wc -l < gen_${PFX}_$TAG.jsonl) seq" >> runs.log
  mkdir -p ids_${PFX}_$TAG; rm -f ids_${PFX}_$TAG/*.ids
  $PY -c "
import json
with open('ids_${PFX}_$TAG/list.txt','w') as L:
    for i, l in enumerate(open('gen_${PFX}_$TAG.jsonl')):
        p = f'$W/ids_${PFX}_$TAG/{i:05d}.ids'; open(p,'w').write(' '.join(map(str, json.loads(l)['ids'])) + '\n'); L.write(p + '\n')"
  rm -rf dump_${PFX}_$TAG
  echo "$(date +%T) dump-${PFX}-$TAG START" >> runs.log
  env CUDA_VISIBLE_DEVICES=0 NINFER_QWEN4_VALIDATION=1 NINFER_QWEN4_TFDUMP=$W/dump_${PFX}_$TAG NINFER_QWEN4_TFDUMP_LAYERS=47 \
      NINFER_QWEN4_TFDUMP_COMPACT=1 NINFER_QWEN4_TFDUMP_TOPK=20 NINFER_QWEN4_TFDUMP_SEQUENCES=1 \
      $DRV score-windows $ART ids_${PFX}_$TAG/list.txt --kv int8 --no-cuda-graph --out dump_${PFX}_$TAG.jsonl > dump_${PFX}_$TAG.log 2>&1
  echo "$(date +%T) dump-${PFX}-$TAG rc=$? $(ls dump_${PFX}_$TAG | grep -c seq) seq" >> runs.log
done
