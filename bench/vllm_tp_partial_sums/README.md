# FP16 partial sums for tensor parallelism in vLLM — measurements

Reproduction material for the vLLM pull request *"Opt-in FP16 partial sums for tensor-parallel row-parallel
layers"* (`VLLM_TP_FP16_PARTIAL_SUMS`). The idea comes from capy-flash, which exchanges its TP partial sums in
FP16/FP32 instead of BF16.

**The finding.** At TP=2 each rank rounds its half of a row-parallel output to BF16 before the all-reduce. Where the
two halves cancel, that rounding dominates the result: on real Qwen3-4B / Qwen3-14B activations, outputs with an
error above 1% of the row RMS occur 4.75x / 5.53x as often as on one GPU. FP16 partial sums (same bytes) bring them
back to the single-GPU level.

## Contents

| path | what |
|---|---|
| `fp16-partial-sums.patch` | the change, against vLLM main `1463501` (7 files incl. the unit test) |
| `run_all.sh` | runs everything below: `run_all.sh <models dir> <out dir>` |
| `scripts/fetch_wiki.py` | evaluation text: 2 long Wikipedia history articles per language (en/it/es/fr, CC BY-SA 4.0) |
| `scripts/make_texts*.py` | token chunks: 1,500-token chunks (292,500 tokens), 4 x 16,000-token chunks, chat prompts |
| `scripts/halfsums2.py` | error at the source: per row-parallel call, the two TP=2 halves in FP32 combined as one GPU / TP=2 BF16 / TP=2 FP16; counts per token with bootstrap CIs, by projection, layer and language |
| `scripts/maxpart.py` | largest partial sum (FP16 range check) |
| `scripts/measure*.py` | vLLM runs: teacher-forced top-20 log-probabilities, greedy generations, decode speed; TP=1 via UVA offload when the model does not fit one GPU |
| `scripts/compare_any.py`, `bivi.py`, `tails.py`, `depth.py` | NLL with paired CIs, KL, top-1 at near-tied positions (McNemar), tails, depth |
| `scripts/repro40725.py`, `corrupt.py` | attempt to reproduce vllm-project/vllm#40725 (not reproduced) |
| `results/` | summaries: `final.log` (all output-level numbers of the PR, final code), `hs2-4b-k2.json`, `hs2-14b.json`, `maxpart.txt` |

Hardware: 2x RTX 5070 Ti 16 GB (SM120, PCIe 5.0 x8, no P2P), Ryzen 9 9900X, torch 2.14, CUDA 13.0. Models:
`Qwen/Qwen3-4B`, `Qwen/Qwen3-14B`, `Qwen/Qwen3-8B-FP8`, Qwen3.8-27B NVFP4 (compressed-tensors), `nvidia/Qwen3.6-35B-A3B-NVFP4`.
Raw per-position log-probabilities (~170 MB per run) are not included; `run_all.sh` regenerates them.
