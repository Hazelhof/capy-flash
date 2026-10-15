<h1 align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/logo/capy-flash-lockup-dark.svg">
  <img alt="capy-flash" src="docs/logo/capy-flash-lockup.svg" width="560">
</picture></h1>

**Qwen3.8-Flash-Next (125B MoE) in NVFP4 on two 16 GB consumer GPUs.**

Against [Strata](https://github.com/Niko1221/Strata) 0.1.44.1 on the same machine (2x RTX 5070 Ti 16 GB, Ryzen 9 9900X, 96 GB DDR5):

| | **capy-flash**, NVFP4 | Strata, same NVFP4 weights | Strata, 3-bit IQ3_S |
|---|---:|---:|---:|
| prefill, 15k-token prompt | **3,435 tok/s** | 2,703 tok/s | 2,906 tok/s |
| prefill, 55k-token prompt | 2,791 tok/s ¹ | 3,821 tok/s | **4,037 tok/s** |
| time to first token, agentic replay | **1.82 s** | 2.42 s | 1.86 s |
| generation, agentic replay | 100.8 tok/s | 78.3 tok/s | **124.1 tok/s** |
| tokens per agentic task run ² | **42.9k** | — | 52.7k |
| time for 20 agentic task runs ² | **4.28 h** | — | 4.32 h |

- **NVIDIA's official NVFP4 experts, used as published**: not re-quantized.
- **Fewer tokens, same finish time**: Strata's 3-bit experts generate faster, but on the same coding tasks they
  write 23 % more tokens, so the agent finishes in the same time.
- **Built for agents**: 20/20 multi-step coding tasks, 0 malformed tool calls in 1,124; OpenAI- and
  Anthropic-compatible API.
- **One engine, two models**: it also serves the dense Qwen3.8-27B, exactly as ninfer-tp2 v0.4.9 did.

¹ Long prompts of new text wait on reading the model's n-gram (PLE) table from disk between chunks; with the rows
cached the 55k prompt runs at 3.4-3.9k tok/s. Overlapping that read with the GPU work is the next change.
² Strata 0.1.43 (0.1.44 did not change its IQ3_S path); 5 multi-step coding tasks x 4 runs, OpenCode. Details below.

**Quick start** (2x RTX 50-series 16 GB, an AVX-512 CPU, ~70 GiB of free RAM):
```bash
hf download Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash --local-dir capy-fn
curl -LO https://github.com/ValerioDolci/capy-flash/releases/download/v0.5.1/capy-flash-v0.5.1-linux-x86_64-sm120a-ubuntu26.04.tar.gz
tar -xzf capy-flash-v0.5.1-*.tar.gz
capy-flash-v0.5.1-*/serve-flash-next.sh capy-fn/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer
```

The routed experts that do not fit in VRAM live in host RAM and are computed on the CPU; everything else runs
tensor-parallel on the two GPUs. capy-flash **is ninfer-tp2, renamed at v0.5.0**: the two-GPU tensor-parallel fork
of [Neroued/ninfer](https://github.com/Neroued/ninfer) (v0.1-v0.4.9 under the old name); [NOTICE](NOTICE) lists every
piece of prior work it builds on.

Status: **v0.5.1** (2026-10-15). Built, tested and measured on one machine (below). Not affiliated with Qwen or NVIDIA.

## What it does
- Tensor parallel over PCIe on two GPUs **without peer-to-peer** (pinned-host mailbox, FP32 partial sums).
- A cache of routed experts in VRAM, partitioned by expert id between the two GPUs; missing experts are computed on
  the CPU (AVX-512 NVFP4 kernel with the GPU's W4A4 activation quantization), inside the captured CUDA graph.
- MTP speculative decoding (K = 3, draft vocabulary shortlist), INT8 KV with an elastic KV / expert-slot budget up to
  262,144 tokens, prefix cache across turns; OpenAI- and Anthropic-compatible server (`ninfer-serve`).
- The same binary still serves the dense **Qwen3.8-27B** (NVFP4, entirely on the two GPUs) exactly like ninfer-tp2 v0.4.9.

## Measured
One machine: AMD Ryzen 9 9900X, 96 GB DDR5-5600, 2x RTX 5070 Ti 16 GB on PCIe 5.0 x8 + x8 (no P2P), Ubuntu, CUDA 13.1.
"GB300" below is the quality reference: NVIDIA's NVFP4 checkpoint served by vLLM on a GB300, same prompts, paired.

**Quality** (serving configuration of this README, 2026-10-15):

| benchmark | capy-flash | GB300 reference | verdict |
|---|---:|---:|---|
| GSM8K, 1,319 questions | 0.9780 | 0.9810 | not worse (McNemar p 0.29; 6 / 2 discordant) |
| MMLU-Pro, 1,008 questions | 0.8482 | 0.8452 | non-inferior (difference +0.30 pt, lower bound -1.14 pt, margin -2.5 pt) |

**Agentic replay** ([bench/capy_flash](bench/capy_flash)): 3 synthetic coding-agent sessions, 53 turns, prompts growing to
~81k tokens with ~90 % prefix reuse, 256 output tokens per turn, temperature 0. Same machine, same day:

| engine | TTFT median (p90) | decode, server side | total |
|---|---:|---:|---:|
| **capy-flash** | **1.82 s (3.33)** | **100.8 tok/s** | **243 s** |
| Strata 0.1.44.1, NVFP4 experts (the same weights) | 2.42 s (5.03) | 78.3 tok/s | 314 s |
| Strata 0.1.44.1, GSQ-RCO IQ3_S experts (25 % smaller, lower quality) | 1.86 s (4.53) | 124.1 tok/s | 236 s |

**Agent tasks**: 5 multi-step coding tasks of a private task bank, 4 runs each, OpenCode with the model's sampling:
20/20 solved, 42.9k generated tokens per run (Strata 0.1.43 IQ3_S on the same tasks: 52.7k, 20/20; the GB300
reference: 26.5k), no malformed tool call in 1,124.

**Prefill** (same prompts on the three engines, server side, no prefix reuse, 2026-10-15): ~3.5k / ~15k / ~55k tokens:
capy-flash 1,941 / 3,435 / 2,791 tok/s; Strata 0.1.44.1 IQ3_S 1,745 / 2,906 / 4,037; Strata NVFP4 1,347 / 2,703 / 3,821.

**Qwen3.8-27B** (same machine, ninfer-tp2 v0.4.9 serving flags, agentic replay at temperature 0): the ninfer-tp2 gate
gives the same golden ids, perplexity (4k 4.385905, 64k 4.133481) and greedy texts (60/60) as v0.4.9; replay TTFT 1.42 s,
decode 171.2 tok/s, total 156 s (v0.4.9: 1.40 s, 172.2 tok/s, 154 s), the same token counts on all 53 turns.

The numbers above were measured on an internal build of 2026-10-14; since then the computation changed only for one PLE table row (see
[development log](docs/capy-flash/development-log.md), T155). The log explains every `Tnn` tag found in the sources.

## Requirements
Linux, CUDA 13.1, two sm_120 GPUs (RTX 50 series) with 16 GB, a CPU with AVX-512 (F/BW/VL, VNNI), and about **70 GiB of
free RAM**: the routed experts are one 64.6 GiB page-locked host allocation. The server checks this at startup and
refuses with the numbers when it does not fit (`NINFER_RAM_GUARD=0` starts anyway).

## Build
```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=OFF -DBUILD_TESTING=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
ninja -C build apps/ninfer-serve
```
Or take the prebuilt tarball of a [release](https://github.com/ValerioDolci/capy-flash/releases) (Ubuntu, `sm_120a`):
it has `serve-flash-next.sh` and `serve-tp2.sh` (27B) with the profiles of this README ([tools/release](tools/release/README-release.md)).

## Model weights
Ready to serve: [`Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash`](https://huggingface.co/Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash)
(126.6 GB, the artifact measured above, with its draft shortlist and `SHA256SUMS`; NVIDIA Open Model License together with
the Qwen Community License of Qwen3.8-Flash-Next):
```
hf download Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash --local-dir <dir>
```
Or convert `nvidia/Qwen3.8-Flash-Next-NVFP4` yourself in one command (artifact + MTP draft shortlist, same result):
```
python tools/capy_flash/convert.py --model <checkpoint dir> --out <dir>
```
It runs `python -m tools.convert --recipe qwen3_8_flash_next_nvfp4_idxbf16_gdnq8 --components text,mtp` (experts NVFP4
as published, indexer BF16, GDN Q8, about 5 minutes on a GPU, 118 GB) and `tools/qwen4_draft_shortlist.py`.

## Serve
```
CUDA_VISIBLE_DEVICES=0,1 build/apps/ninfer-serve <dir>/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer \
  --host 127.0.0.1 --port 8098 --model-id capy-flash --tp 2 --devices 0,1 --spec mtp --draft-tokens 3 \
  --kv-dtype int8 --elastic --max-context 262144 --kv-capacity 262144 --max-concurrency 1
```
The CPU expert path quantizes activations like the GPU path (W4A4, on by default; `NINFER_HOST_MOE_A4=0` restores
the older W4A8, which made the output depend on which experts happened to be in VRAM). `--turn-anchors` is 0 by
default on Flash-Next, which saves one prefill pass per agentic turn (about 0.8 s here); the cost is that a cancelled or
edited turn re-reads the whole conversation (`--turn-anchors 1` restores the anchor). Models that sit entirely on the
GPU (Qwen3.8-27B) keep the anchor by default: there the extra pass costs nothing measurable. Other switches, diagnostics and dumps:
[docs/capy-flash/diagnostics.md](docs/capy-flash/diagnostics.md); general server options: [docs/serving.md](docs/serving.md).

## Tests
`ctest --test-dir build` runs the unit tests (187). The Engine integration tests run when their artifacts are set:
`NINFER_TEST_QWEN4_ARTIFACT` (a small synthetic Flash-Next artifact with the real layer geometry, made by
`tools/capy_flash/make_test_artifact.sh <checkpoint dir> <out dir>`, CPU only) and `NINFER_TEST_ARTIFACT` (a
Qwen3.8-27B artifact, for the tests inherited from ninfer-tp2). For the 27B, `tools/tp2/gate.sh` is the reference run:
it sets the artifacts, runs the two-device integration tests and excludes the seven single-GPU tests that need other
artifact kinds (DFlash, MoE, vision workspace; `GATE_CTEST_EXCLUDE`), then checks golden ids, perplexity and greedy
texts against a recorded reference ([tools/tp2/README.md](tools/tp2/README.md)).

## Known limits
- One request at a time (`--max-concurrency 1`); tested only on the machine above.
- Generated tokens per agent task are about 1.6x the GB300 reference (above): the remaining gap with the datacenter
  stack is length of reasoning, not correctness.
- Output is not bit-reproducible run to run: which experts sit in VRAM (GPU path) or not (CPU path) depends on timing,
  and long streamed prefills differ in the last bits; quality is measured with that in place.
- Decode is bound by host-RAM bandwidth for the experts computed on the CPU (about 270-300 per MTP window).

## License
Apache License 2.0 ([LICENSE](LICENSE)); attribution in [NOTICE](NOTICE). The original ninfer-tp2 README is in
[docs/README-ninfer-tp2.md](docs/README-ninfer-tp2.md).
