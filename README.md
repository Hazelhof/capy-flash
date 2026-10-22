<h1 align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/logo/capy-flash-lockup-dark.svg">
  <img alt="capy-flash" src="docs/logo/capy-flash-lockup.svg" width="560">
</picture></h1>

**Qwen3.8-Flash-Next (125B MoE) in NVFP4 on two 16 GB consumer GPUs, built for agents.**

Against [Strata](https://github.com/Niko1221/Strata) 0.1.45 on the same machine, same day (2x RTX 5070 Ti 16 GB,
Ryzen 9 9900X, 96 GB DDR5, full clocks):

| | **capy-flash**, NVFP4 | Strata, same NVFP4 weights | Strata, 3-bit IQ3_S (tuned) |
|---|---:|---:|---:|
| agentic replay, 53 turns up to 81k tokens: total time | **177 s** | 268 s | 197 s |
| agentic replay: decode, server side (median) | **155 tok/s** | 98 tok/s | 151 tok/s |
| agentic replay: time to first token, median (p90) | **1.48 s (2.62)** | 2.35 s (4.17) | 1.60 s (3.69) |
| prefill, ~15k-token prompt | **4,615 tok/s** | 3,513 tok/s | 3,811 tok/s |
| prefill, ~55k-token prompt | 4,828 tok/s | 4,849 tok/s | **5,233 tok/s** |
| LocalMaxxing reasoning-v1 (short prompt, 512 tokens) | 178 tok/s | 119 tok/s | **195 tok/s** |

- **NVIDIA's official NVFP4 experts, used as published**: not re-quantized. Strata's IQ3_S experts are 26 % smaller
  (lower quality: KL 0.144 vs 0.070 for NVFP4, as Strata reports), which is why it leads on the short benchmark.
- **Decode stays flat with context**: 168 / 165 / 171 / 165 tok/s with 0 / 14k / 61k / 118k tokens of prompt.
- **Built for agents**: idle conversations move to RAM and give their VRAM back to the expert cache, a continued
  conversation keeps the experts its next answer needs, prefix cache across turns; OpenAI- and Anthropic-compatible API.
- **One engine, two models**: it also serves the dense **Qwen3.8-27B** (prefill +61-65 % over v0.5.4 at 4k-16k, same
  tokens; 175 / 326 / 560 / 927 tok/s aggregate at 1 / 2 / 4 / 8 concurrent requests).
- **Vision on Flash-Next** (opt-in, `--vision` with a vision artifact).

**New in v0.7: start small, grow later, same engine.**
- **Flash-Next on 64 GB of RAM, with one or two GPUs** (`--model flash-next-64gb`): ISTA-DASLab's GSQ-RCO i-quant experts
  (48 GiB in RAM instead of 65). Two GPUs: the fastest way to run it here, LocalMaxxing reasoning-v1 **239 tok/s**,
  agentic replay decode **182 tok/s** (NVFP4 172 / 155). **One 16 GB GPU** (`NINFER_DEVICES=0`, `--profile
  flash-next-1gpu`): **121 tok/s** (LocalMaxxing verified run), replay decode 104. Quality on par with NVFP4 (GSM8K, MMLU-Pro).
- **Qwen3.8-27B on one 16 GB card** (`--model 27b-16gb`, 3-bit GSQ weights, 64k context, vision): **139 tok/s** decode
  with MTP on an 8k-token prompt (two cards: 230) and 3,401 tok/s prefill at 4k. **It costs quality**: GSM8K on par,
  MMLU-Pro about 0.72 against 0.78 for the NVFP4 27B ([below](#measured)).
- **AVX2 CPUs** (Zen 2/3, Intel client) and **Windows/WSL2** with its page-locked memory limit start Flash-Next too.
- Empty-reply guards on by default for every model; boards with P2P overlap the prefill exchanges on the direct copy
  (+29-45 % measured by @tzolkincz); a context-cache accounting error fails one request instead of the server.

**Quick start** (2x RTX 50-series 16 GB, an AVX2 or AVX-512 CPU, ~75 GiB of free RAM, 135 GB of disk; smaller
machines: `--model flash-next-64gb` or `--model 27b-16gb`, below):
```bash
curl -fsSL https://raw.githubusercontent.com/ValerioDolci/capy-flash/main/install.sh | bash -s -- --check   # can this machine run it?
curl -fsSL https://raw.githubusercontent.com/ValerioDolci/capy-flash/main/install.sh | bash               # release + model into ~/capy-flash
~/capy-flash/current/serve-flash-next.sh ~/capy-flash/models/flash-next/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"capy-flash","messages":[{"role":"user","content":"What is 17*23?"}]}'                # from another shell
```
The installer checks GPUs, driver, AVX2/AVX-512, RAM and disk and says what to fix, verifies every download, and prints
a test request; `--model flash-next-64gb` installs the 64 GB edition, `--model 27b` Qwen3.8-27B for two GPUs,
`--model 27b-16gb` the 3-bit 27B for one GPU, `--dir` picks another place. By hand: download a
[release](https://github.com/ValerioDolci/capy-flash/releases) tarball and `hf download Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash`.

The routed experts that do not fit in VRAM live in host RAM and are computed on the CPU; everything else runs
tensor-parallel on the two GPUs. capy-flash **is ninfer-tp2, renamed at v0.5.0**: the two-GPU tensor-parallel fork
of [Neroued/ninfer](https://github.com/Neroued/ninfer) (v0.1-v0.4.9 under the old name); [NOTICE](NOTICE) lists every
piece of prior work it builds on.

Status: **v0.7.1** (2026-10-22). Built, tested and measured on one machine (below). Not affiliated with Qwen or NVIDIA.

## What it does
- Tensor parallel over PCIe on two GPUs **without peer-to-peer** (pinned-host mailbox, FP32 partial sums); with P2P
  (for example two boards behind a PCIe switch) the prefill exchanges take the direct copy automatically (v0.6).
- A cache of routed experts in VRAM, partitioned by expert id between the two GPUs; missing experts are computed on
  the CPU (NVFP4 kernels in AVX-512 VNNI or AVX2, chosen at startup and identical bit for bit, with the GPU's W4A4
  activation quantization), inside the captured CUDA graph.
- MTP speculative decoding (K = 3, draft vocabulary shortlist), INT8 KV with an elastic KV / expert-slot budget up to
  262,144 tokens, prefix cache across turns; OpenAI- and Anthropic-compatible server (`ninfer-serve`).
- The same binary still serves the dense **Qwen3.8-27B** (NVFP4, entirely on the two GPUs) exactly like ninfer-tp2 v0.4.9.
- Context cache for agents (v0.5.4): a request reserves its prompt plus a 4,096-token window of KV and grows it as the
  answer grows (a large `max_tokens` no longer pushes other conversations out of VRAM), and with two GPUs the cache can
  keep evicted conversations in RAM (`--host-kv-mib`, `--host-state-slots`; on in `--profile 27b` with 8 GiB). On
  Flash-Next (v0.6) the KV of idle conversations moves to RAM (`--profile flash-next`: 4 GiB) and the VRAM it frees goes
  back to expert slots at once. On an agentic replay of Qwen3.8-27B (3 interleaved sessions
  and 11 subagents, production flags, 4 lanes): median time to first token 2.14 → 0.98 s, prompt tokens served from
  cache 67 → 78 %, prompt tokens recomputed −35 %, decode unchanged.
- Expert residency that follows the conversation (v0.6): a prompt that continues a cached conversation keeps the
  experts its next answer needs, a new conversation only updates their statistics; short prompts stream their experts
  to the GPUs from 256 tokens (`--stream-min-tokens`). The post-prompt slowdown of v0.5.x (~250 ms per turn) is gone.
- MTP draft head re-distilled on this engine's own output (v0.6): +18 % decode in Italian, +4-8 % in English and code.
- Vision on Flash-Next (v0.6, opt-in): `--vision` with an artifact converted by `tools/capy_flash/convert.py --vision`
  (Q8 tower, 0.45 GiB of VRAM, ~4 % decode); without `--vision` nothing changes.
- NVIDIA's FP4-Local checkpoint runs too (recipe `qwen3_8_flash_next_fp4_local`), ~19 % slower than the default artifact.
- GGUF i-quant experts (v0.7): IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL and Q2_0 blocks with variable-size expert slots
  per layer, GPU kernels from ggml (via fngine / ninfer-all), CPU kernels in AVX-512 VNNI and AVX2; the 64 GB edition of
  Flash-Next takes its experts verbatim from ISTA-DASLab's GSQ-RCO GGUF.
- 3-bit GSQ weights for the dense 27B (v0.7, `q3_g128_fp16`): tensor-core kernels with int8 activations in the prefill
  (`NINFER_Q3_A8=0` for 16-bit), `--profile 27b-16gb` for one 16 GB card.
- Page-locked memory limit (v0.7): when the system grants less pinned memory than the experts need (Windows/WSL2 caps it
  at about half the RAM), the layers that don't fit stay in pageable RAM: the CPU reads them in place, the GPUs copy
  them through a pinned staging ring (`--host-pinned-max-gib` sets the cap by hand). Same tokens as fully pinned.
- Empty-reply guards (v0.7, every model, on by default): a terminal token drawn right after `<think>` is redrawn, a reply
  that ends inside its reasoning is closed and continued once (`--no-reasoning-eos-guard`, `--no-reasoning-close-retry`).

## Measured
One machine: AMD Ryzen 9 9900X, 96 GB DDR5-5600, 2x RTX 5070 Ti 16 GB on PCIe 5.0 x8 + x8 (no P2P), Ubuntu, CUDA 13.1.
"GB300" below is the quality reference: NVIDIA's NVFP4 checkpoint served by vLLM on a GB300, same prompts, paired.

**Quality** (v0.6 release candidate, 2026-10-18, serving configuration of this README):

| benchmark | capy-flash | GB300 reference | verdict |
|---|---:|---:|---|
| GSM8K, 1,319 questions | 0.9750 | 0.9810 | 9 / 1 discordant, McNemar p 0.02 (v0.5.3: 0.9765, p 0.07) |
| MMLU-Pro, 1,008 questions | 0.8413 | 0.8452 | non-inferior (difference -0.40 pt, lower bound -1.85 pt, margin -2.5 pt) |

The final binary was checked on the first 330 / 252 questions (GSM8K 0.979, MMLU-Pro 0.841). Output is not
bit-reproducible run to run (which experts sit in VRAM depends on timing), so two runs of the same build differ by a
few questions.

**Agentic replay** ([bench/capy_flash](bench/capy_flash)): 3 synthetic coding-agent sessions, 53 turns, prompts growing to
~81k tokens with ~90 % prefix reuse, 256 output tokens per turn, temperature 0. Same machine, same day (2026-10-18),
full clocks: table at the top. v0.6 against the build before the v0.6 work (2026-10-17): total 212 → 177 s, decode
118 → 155 tok/s.

**Agent tasks** (2026-10-14 build): 5 multi-step coding tasks of a private task bank, 4 runs each, OpenCode with the
model's sampling: 20/20 solved, 42.9k generated tokens per run (Strata 0.1.43 IQ3_S on the same tasks: 52.7k, 20/20;
the GB300 reference: 26.5k), no malformed tool call in 1,124.

**Prefill** (same prompts on the three engines, server side, no prefix reuse, ~3.5k / ~15k / ~55k tokens): capy-flash
v0.6 2,422 / 4,615 / 4,828 tok/s; Strata 0.1.45 IQ3_S 2,216 / 3,811 / 5,233; Strata NVFP4 1,519 / 3,513 / 4,849.
A cold 382-token prompt answers in 1.35 s (2.5-2.9 s with the previous 512-token streaming threshold).

**Start**: a cold start of the Flash-Next server takes 26-29 s here (measured on v0.5.3), most of it reading the
64.6 GiB of experts from disk.

**Qwen3.8-27B** (`--profile 27b`, MTP; 280 prompts of three languages, server side): the same tokens as v0.5.4 on 280 of
280 answers (and with DFlash2), decode +5 % (Italian 168 → 176 tok/s), prefill at 4k / 16k / 64k 5,777 / 5,585 /
4,352 → 9,343 / 9,234 / 6,287 tok/s with MTP and 5,719 / 5,530 / 4,316 → 9,205 / 9,052 / 6,198 with DFlash2.
Concurrent decode (NInfer's `decode-saturation` suite, MTP3, 8,192 tokens per request): 175 / 326 / 560 / 927 tok/s
aggregate at 1 / 2 / 4 / 8 requests.

**v0.7** (2026-10-22, same machine, full clocks): the NVFP4 Flash-Next and 27B artifacts run as in v0.6 (27B: the same
tokens on 280 of 280 prompts; Flash-Next decode within ±2 % on six prompt sets; GSM8K 0.985 and MMLU-Pro 0.829 on the
first 330 / 252 questions, paired with v0.6: no difference).

| v0.7, new artifacts | Flash-Next 64 GB, two GPUs | Flash-Next 64 GB, one GPU | Qwen3.8-27B 3-bit, one GPU | Qwen3.8-27B 3-bit, two GPUs |
|---|---:|---:|---:|---:|
| LocalMaxxing reasoning-v1 / agentic replay decode | 239 / 182 tok/s | 121 / 104 tok/s | — | — |
| prefill 4k / 16k / 64k | 2,117 / 3,305 / 3,426 | 1,888 / 2,145 / 2,257 | 3,401 / 3,211 / — | 5,787 / 5,567 / 4,304 |
| decode, 8k real-text prompt, MTP | — | — | 139 tok/s | 230 tok/s |
| GSM8K (vs GB300 NVFP4 reference) | 0.982 (0.985), 330 q. | same weights | 0.968 (0.968), 1,319 q. | |
| MMLU-Pro (vs GB300 NVFP4 reference) | 0.837 (0.849), 252 q. | same weights | **0.69-0.71, ~0.72 with the guards (0.78)**, 252 q. | |

The 64 GB edition peaks at 55.2 GiB of RSS inside a 64 GiB memory cgroup without swap (57 GiB used with one GPU);
speeds were measured on the 96 GB machine. One GPU keeps only one conversation checkpoint in VRAM, so v0.7.1's
`--profile flash-next-1gpu` keeps 8 in RAM (+0.66 GiB): on the agentic replay 83 % of the prompt tokens come from the
cache, 1.9 s to the first token (median) and 276 s for the whole replay (v0.7.0: 34 %, 13.3 s, 858 s). The 3-bit 27B loses MMLU-Pro
points mostly because more answers run out of their 8,192 tokens while reasoning; the v0.7 empty-reply guards bring
back the ones that stopped inside their reasoning (9 of 9, all correct: about +3.6 points, estimated on those
questions). With two cards, the NVFP4 27B is the better choice. At the eco clocks of 2026-10-19 the one-card 27B
decoded 105 tok/s where llama.cpp (ISTA's GSQ-RCO IQ3_S with its MTP drafting, same card and prompt) did 66-67.

See the [development log](docs/capy-flash/development-log.md), which explains every `Tnn` tag found in the sources.

## Requirements
Linux, CUDA 13.1, two sm_120 GPUs (RTX 50 series) with 16 GB, a CPU with AVX-512 (F/BW/VL, VNNI) or AVX2 + FMA + F16C
(Zen 2/3, Intel client cores; `NINFER_HOST_ISA` in [diagnostics](docs/capy-flash/diagnostics.md)), and about **75 GiB of
free RAM** (a 96 GB machine): the routed experts are one 64.6 GiB page-locked host allocation, and `--profile flash-next`
also keeps the token embedding and up to 4 GiB of idle conversations in RAM. The 27B needs neither of these CPU features nor that RAM.
**Smaller machines (v0.7):** the 64 GB edition of Flash-Next needs about 56 GiB of free RAM (48 GiB of experts) and runs
on one or two 16 GB cards; the 3-bit 27B runs on one 16 GB card. Tested here on Linux; users run the 27B on WSL2 and on Windows with Docker. The server checks this at startup and
refuses with the numbers when it does not fit (`NINFER_RAM_GUARD=0` starts anyway).

## Build
```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=OFF -DBUILD_TESTING=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
ninja -C build apps/ninfer-serve
```
Or take the prebuilt tarball of a [release](https://github.com/ValerioDolci/capy-flash/releases) (Ubuntu, `sm_120a`):
it has `serve-flash-next.sh`, `serve-tp2.sh` (27B, two GPUs) and `serve-27b-16gb.sh` (27B 3-bit, one GPU) with the profiles of this README ([tools/release](tools/release/README-release.md)).

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

With images (`--vision`, see below): [`Feyd89/Qwen3.8-Flash-Next-NVFP4-Vision-capy-flash`](https://huggingface.co/Feyd89/Qwen3.8-Flash-Next-NVFP4-Vision-capy-flash)
(the same artifact plus the Q8 vision tower, capy-flash v0.6.0 or newer). The previous text artifact, with NVIDIA's MTP
head, stays available as revision `v0.5.4` of the first repository (`hf download ... --revision v0.5.4`).

v0.7: [`Feyd89/Qwen3.8-Flash-Next-GSQ-RCO-capy-flash`](https://huggingface.co/Feyd89/Qwen3.8-Flash-Next-GSQ-RCO-capy-flash)
(102 GB, the 64 GB edition: ISTA-DASLab's GSQ-RCO experts, the rest as above) and
[`Feyd89/Qwen3.8-27B-GSQ-3bit-capy-flash`](https://huggingface.co/Feyd89/Qwen3.8-27B-GSQ-3bit-capy-flash) (12.5 GB,
ISTA-DASLab's 3-bit GSQ 27B plus the official MTP layer; convert it yourself with the recipe `qwen3_8_27b_gsq3`).

## Serve
```
build/apps/ninfer-serve <dir>/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer --profile flash-next --port 8098 --model-id capy-flash
```
`--profile flash-next` is the configuration measured above (`--tp 2 --devices 0,1 --kv-dtype int8 --elastic
--max-context 262144 --max-concurrency 1 --spec mtp --draft-tokens 3`); `--profile 27b` is the ninfer-tp2 one for
Qwen3.8-27B (with Vision; `--no-vision` for artifacts without it); `--profile 27b-16gb` runs the 3-bit 27B on one card
(int8 KV, 65,536 tokens, vision, 13.9 GB of VRAM). For the 64 GB edition of Flash-Next add `--host-kv-mib 1024
--host-state-slots 2` to `--profile flash-next` (smaller RAM caches), or use `--profile flash-next-1gpu` on one GPU. Flags given on the command line override the
profile's, in any order (`ninfer-serve --help` lists both). To keep it running: a systemd unit and a llama-swap entry
in [examples/serving](examples/serving).

**Images (opt-in).** Flash-Next is multimodal: convert with `--vision` (`python tools/capy_flash/convert.py --model
<checkpoint dir> --out <dir> --vision`, i.e. `tools.convert ... --components text,vision,mtp`; the Text and MTP parts
come from the same conversion as the text-only artifact, the tower is stored Q8; ready-made on
[Hugging Face](https://huggingface.co/Feyd89/Qwen3.8-Flash-Next-NVFP4-Vision-capy-flash)) and start with `--profile flash-next --vision`
(tower on GPU 0, `--vision-device 1` moves it, `--max-vision-tokens N` caps one image). Images and videos are accepted
in the chat APIs and a conversation with an image resumes its cached prefix. Measured here (`--max-vision-tokens
4096`): tower 0.45 GiB on its GPU, 4 expert slots per layer fewer (66 to 62), decode about 4 % slower on short prompts,
prefill unchanged; its merged embeddings are within 0.09 (relative) of a BF16 PyTorch reference. Without `--vision` the tower is not loaded and nothing changes.

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
`ctest --test-dir build` runs the unit tests (208). The Engine integration tests run when their artifacts are set:
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
