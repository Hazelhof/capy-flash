<h1 align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/logo/capy-flash-lockup-dark.svg">
  <img alt="capy-flash" src="docs/logo/capy-flash-lockup.svg" width="560">
</picture></h1>

**Qwen3.8-Flash-Next (125B MoE) in NVFP4 on two 16 GB consumer GPUs.**

Against [Strata](https://github.com/Niko1221/Strata) 0.1.44.1 on the same machine (2x RTX 5070 Ti 16 GB, Ryzen 9 9900X, 96 GB DDR5):

| | **capy-flash**, NVFP4 | Strata, same NVFP4 weights | Strata, 3-bit IQ3_S |
|---|---:|---:|---:|
| prefill, 15k-token prompt | **3,778 tok/s** | 2,703 tok/s | 2,906 tok/s |
| prefill, 55k-token prompt | 3,977 tok/s ¹ | 3,821 tok/s | **4,037 tok/s** |
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

¹ Same prompts as the Strata runs, new text each time (the n-gram (PLE) table rows come from disk). Since v0.5.2 the
next chunk's rows are read while the GPUs run the current one; v0.5.3 keeps the rows' multipliers in RAM (the first
chunk of ~15k new tokens waits 0.3-0.5 s instead of 0.8-1.3 s) and computes the router on tensor cores.
² Strata 0.1.43 (0.1.44 did not change its IQ3_S path); 5 multi-step coding tasks x 4 runs, OpenCode. Details below.

**Quick start** (2x RTX 50-series 16 GB, an AVX-512 CPU, ~70 GiB of free RAM, 135 GB of disk):
```bash
curl -fsSL https://raw.githubusercontent.com/ValerioDolci/capy-flash/main/install.sh | bash -s -- --check   # can this machine run it?
curl -fsSL https://raw.githubusercontent.com/ValerioDolci/capy-flash/main/install.sh | bash               # release + model into ~/capy-flash
~/capy-flash/current/serve-flash-next.sh ~/capy-flash/models/flash-next/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"capy-flash","messages":[{"role":"user","content":"What is 17*23?"}]}'                # from another shell
```
The installer checks GPUs, driver, AVX-512, RAM and disk and says what to fix, verifies every download, and prints
a test request; `--model 27b` installs Qwen3.8-27B instead, `--dir` picks another place. By hand: download a
[release](https://github.com/ValerioDolci/capy-flash/releases) tarball and `hf download Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash`.

The routed experts that do not fit in VRAM live in host RAM and are computed on the CPU; everything else runs
tensor-parallel on the two GPUs. capy-flash **is ninfer-tp2, renamed at v0.5.0**: the two-GPU tensor-parallel fork
of [Neroued/ninfer](https://github.com/Neroued/ninfer) (v0.1-v0.4.9 under the old name); [NOTICE](NOTICE) lists every
piece of prior work it builds on.

Status: **v0.5.4** (2026-10-16). Built, tested and measured on one machine (below). Not affiliated with Qwen or NVIDIA.

## What it does
- Tensor parallel over PCIe on two GPUs **without peer-to-peer** (pinned-host mailbox, FP32 partial sums).
- A cache of routed experts in VRAM, partitioned by expert id between the two GPUs; missing experts are computed on
  the CPU (AVX-512 NVFP4 kernel with the GPU's W4A4 activation quantization), inside the captured CUDA graph.
- MTP speculative decoding (K = 3, draft vocabulary shortlist), INT8 KV with an elastic KV / expert-slot budget up to
  262,144 tokens, prefix cache across turns; OpenAI- and Anthropic-compatible server (`ninfer-serve`).
- The same binary still serves the dense **Qwen3.8-27B** (NVFP4, entirely on the two GPUs) exactly like ninfer-tp2 v0.4.9.
- Context cache for agents (v0.5.4): a request reserves its prompt plus a 4,096-token window of KV and grows it as the
  answer grows (a large `max_tokens` no longer pushes other conversations out of VRAM), and with two GPUs the cache can
  keep evicted conversations in RAM (`--host-kv-mib`, `--host-state-slots`; on in `--profile 27b` with 8 GiB, off on
  Flash-Next, whose experts already pin most of the RAM). On an agentic replay of Qwen3.8-27B (3 interleaved sessions
  and 11 subagents, production flags, 4 lanes): median time to first token 2.14 → 0.98 s, prompt tokens served from
  cache 67 → 78 %, prompt tokens recomputed −35 %, decode unchanged.

## Measured
One machine: AMD Ryzen 9 9900X, 96 GB DDR5-5600, 2x RTX 5070 Ti 16 GB on PCIe 5.0 x8 + x8 (no P2P), Ubuntu, CUDA 13.1.
"GB300" below is the quality reference: NVIDIA's NVFP4 checkpoint served by vLLM on a GB300, same prompts, paired.

**Quality** (serving configuration of this README, v0.5.3, 2026-10-16):

| benchmark | capy-flash | GB300 reference | verdict |
|---|---:|---:|---|
| GSM8K, 1,319 questions | 0.9765 | 0.9810 | not worse (McNemar p 0.070; 7 / 1 discordant) |
| MMLU-Pro, 1,008 questions | 0.8452 | 0.8452 | non-inferior (difference 0.00 pt, lower bound -1.37 pt, margin -2.5 pt) |

v0.5.3 computes the router of prompt chunks on tensor cores, which sums in another order and can pick another expert
where two are tied; with the previous router (`NINFER_QWEN4_ROUTER_MMA=0`) the same gate gave 0.9780 / 0.8482 on the 2026-10-14 build.

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

**Prefill** (same prompts on the three engines, server side, no prefix reuse): ~3.5k / ~15k / ~55k tokens:
capy-flash v0.5.3 1,985 / 3,778 / 3,977 tok/s (2026-10-16; v0.5.2: 1,856 / 3,302 / 3,715; v0.5.1: 1,941 / 3,435 /
2,791); Strata 0.1.44.1 IQ3_S 1,745 / 2,906 / 4,037; Strata NVFP4 1,347 / 2,703 / 3,821 (2026-10-15).

**Start**: a cold start of the Flash-Next server takes 26-29 s here (v0.5.2: 56-141 s), most of it reading the 64.6 GiB
of experts from disk.

**Qwen3.8-27B** (same machine, ninfer-tp2 v0.4.9 serving flags, agentic replay at temperature 0): the ninfer-tp2 gate
gives the same golden ids, perplexity (4k 4.385905, 64k 4.133481) and greedy texts (60/60) as v0.4.9; replay TTFT 1.42 s,
decode 171.2 tok/s, total 156 s (v0.4.9: 1.40 s, 172.2 tok/s, 154 s), the same token counts on all 53 turns.

Quality, prefill and start were measured on v0.5.3; the agentic replay and agent tasks on an internal build of 2026-10-14,
whose computation differs from v0.5.3 in one PLE table row (T155) and in the prompt router (T183, above); see the
[development log](docs/capy-flash/development-log.md), which explains every `Tnn` tag found in the sources.

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
build/apps/ninfer-serve <dir>/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer --profile flash-next --port 8098 --model-id capy-flash
```
`--profile flash-next` is the configuration measured above (`--tp 2 --devices 0,1 --kv-dtype int8 --elastic
--max-context 262144 --max-concurrency 1 --spec mtp --draft-tokens 3`); `--profile 27b` is the ninfer-tp2 one for
Qwen3.8-27B (with Vision; `--no-vision` for artifacts without it). Flags given on the command line override the
profile's, in any order (`ninfer-serve --help` lists both). To keep it running: a systemd unit and a llama-swap entry
in [examples/serving](examples/serving).

**Docker.** [docker/Dockerfile.release](docker/Dockerfile.release) packs a release tarball (1.2 GB, builds in about a
minute; the root [Dockerfile](Dockerfile) compiles from source instead). It needs the NVIDIA Container Toolkit:
```
docker build -f docker/Dockerfile.release -t capy-flash .
docker run --gpus all -p 8080:8080 -v ~/capy-flash/models:/models capy-flash \
  /models/flash-next/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer --profile flash-next --host 0.0.0.0
```
Tested with rootless Podman and the two GPUs passed through (Flash-Next loads its 64.6 GiB of page-locked experts
without extra limits).
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
- With several requests decoding, an answer that outgrows its KV window when the GPUs are full first moves idle cached
  conversations to RAM (or drops them without a RAM tier); only if that cannot free a page does it end early with
  `finish_reason: length` (the request log counts it). `--kv-lease-window 0` restores the full reservation of v0.5.3.
- Moving a conversation between RAM and the GPUs shares PCIe with the two GPUs' exchange: decode of other requests
  is slower while a move is in flight (no difference otherwise).
- Generated tokens per agent task are about 1.6x the GB300 reference (above): the remaining gap with the datacenter
  stack is length of reasoning, not correctness.
- Output is not bit-reproducible run to run: which experts sit in VRAM (GPU path) or not (CPU path) depends on timing,
  and long streamed prefills differ in the last bits; quality is measured with that in place.
- Decode is bound by host-RAM bandwidth for the experts computed on the CPU (about 270-300 per MTP window).

## License
Apache License 2.0 ([LICENSE](LICENSE)); attribution in [NOTICE](NOTICE). The original ninfer-tp2 README is in
[docs/README-ninfer-tp2.md](docs/README-ninfer-tp2.md).
