#!/bin/bash
# Reproduce the measurements of the vLLM PR "Opt-in FP16 partial sums for tensor-parallel row-parallel layers".
# Needs: vLLM with the PR applied, 2 GPUs (TP=2) of which one fits each single-GPU reference, HF transformers.
# Usage: run_all.sh <models dir> <out dir>      (models: qwen3-4b, qwen3-14b, qwen3-8b-fp8, as HF snapshots)
set -u
M=$1; R=$2; S=$(cd "$(dirname "$0")/scripts" && pwd); P=${PYTHON:-python}; mkdir -p $R
$P $S/fetch_wiki.py $R/wiki.json
$P $S/make_texts.py $M/qwen3-4b $R/texts.json $R/wiki.json
$P $S/make_texts_big.py $M/qwen3-4b $R/big.json $R/long.json $R/wiki.json
# 1. error at the source (HF transformers, one GPU / device_map=auto)
CUDA_VISIBLE_DEVICES=0 $P $S/halfsums2.py $M/qwen3-4b $R/big.json $R/hs2-4b.json 32
$P $S/halfsums2.py $M/qwen3-14b $R/big.json $R/hs2-14b.json 32
$P $S/maxpart.py $M/qwen3-14b $R/big.json 16
# 2. output-level, Qwen3-4B BF16: TP=1, TP=2 BF16, TP=2 FP16, FP32 reference (TP=2)
for cfg in "0 1 0 bfloat16 tp1" "0,1 2 0 bfloat16 tp2bf16" "0,1 2 1 bfloat16 tp2fp16" "0,1 2 0 float32 fp32"; do set -- $cfg
  CUDA_VISIBLE_DEVICES=$1 VLLM_TP_FP16_PARTIAL_SUMS=$3 DTYPE=$4 MAXLEN=4096 MBT=1024 LOGPROBS_ONLY=1 GPU_UTIL=0.70 \
    $P $S/measure_big.py $M/qwen3-4b $2 $R/big.json $R/4b-$5.json
done
$P $S/compare_any.py $R/4b-tp1.json $R/4b-tp2bf16.json $R/4b-tp2fp16.json
$P $S/compare_any.py $R/4b-fp32.json $R/4b-tp1.json $R/4b-tp2bf16.json $R/4b-tp2fp16.json
$P $S/bivi.py $R/4b-tp1.json $R/4b-tp2bf16.json $R/4b-tp2fp16.json
$P $S/tails.py $R/4b-tp1.json $R/4b-tp2bf16.json $R/4b-tp2fp16.json
# 3. FP8 (block FP8 needs a covered kernel: --linear-backend cutlass)
for cfg in "0 1 0 tp1" "0,1 2 0 tp2bf16" "0,1 2 1 tp2fp16"; do set -- $cfg
  CUDA_VISIBLE_DEVICES=$1 VLLM_TP_FP16_PARTIAL_SUMS=$3 LINEAR_BACKEND=cutlass DTYPE=bfloat16 MAXLEN=4096 LOGPROBS_ONLY=1 GPU_UTIL=0.80 \
    $P $S/measure_offload.py $M/qwen3-8b-fp8 $2 $R/big.json $R/fp8-$4.json
done
$P $S/compare_any.py $R/fp8-tp1.json $R/fp8-tp2bf16.json $R/fp8-tp2fp16.json
$P $S/bivi.py $R/fp8-tp1.json $R/fp8-tp2bf16.json $R/fp8-tp2fp16.json
