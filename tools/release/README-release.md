# capy-flash — prebuilt binaries

capy-flash (formerly ninfer-tp2) serves **Qwen3.8-Flash-Next** (MoE, NVFP4 experts, CPU offload) and **Qwen3.8-27B**
on two 16 GB Blackwell GeForce boards; since v0.7 also Flash-Next on 64 GB of RAM and the 27B (3-bit) on one board. Source and documentation: https://github.com/ValerioDolci/capy-flash.
`COMMIT` records the exact commit.

## Contents

| Path | What |
|---|---|
| `bin/ninfer-serve` | OpenAI- and Anthropic-compatible server |
| `bin/ninfer` | command-line generation |
| `bin/ninfer-perplexity` | perplexity evaluator |
| `lib/libcudart.so.13` | the CUDA runtime the binaries link (NVIDIA redistributable) |
| `serve-flash-next.sh` | launcher with the Qwen3.8-Flash-Next profile |
| `serve-tp2.sh` | launcher with the Qwen3.8-27B two-GPU production profile (as in ninfer-tp2) |
| `serve-27b-16gb.sh` | launcher for the 3-bit Qwen3.8-27B on one 16 GB board (`--profile 27b-16gb`) |
| `SHA256SUMS` | digests of every file above; check with `sha256sum -c SHA256SUMS` |

## Requirements

- Linux x86_64 with the distribution named in the tarball (the binaries link its FFmpeg and libcurl:
  `libavcodec`, `libavformat`, `libavutil`, `libswscale`, `libcurl4`). On another distribution
  build from source or use the Dockerfile.
- An NVIDIA driver that supports CUDA 13.1 (R580 or newer). No CUDA toolkit is needed.
- Two `sm_120` boards with 16 GB for `--tp 2` (one for the 3-bit 27B) (validated on 2x RTX 5070 Ti without P2P; with P2P, for example through
  a PCIe switch, the prefill exchanges use the direct copy automatically).
- **Qwen3.8-Flash-Next** also needs a CPU with AVX-512 (F/BW/VL, VNNI) or AVX2 + FMA + F16C (the expert kernels of
  both give the same results bit for bit) and about **75 GiB of free RAM**: the routed
  experts that stay on the host are one 64.6 GiB page-locked allocation, `--profile flash-next` keeps the token
  embedding and up to 4 GiB of idle conversations' KV in RAM as well, and the server refuses to start without room for
  the experts (`NINFER_RAM_GUARD=0` skips the check). In practice a 96 GB machine; the 64 GB edition (GSQ-RCO
  experts, 48 GiB) needs about 56 GiB and fits a 64 GB machine.

## Run

```bash
tar -xzf capy-flash-*.tar.gz && cd capy-flash-*/
sha256sum -c SHA256SUMS
./serve-flash-next.sh /path/to/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer          # Qwen3.8-Flash-Next
./serve-flash-next.sh /path/to/vision-artifact.ninfer --vision                    # Flash-Next with images (opt-in)
./serve-flash-next.sh /path/to/qwen3_8_flash_next_gsqrco_mtpB.ninfer --host-kv-mib 1024 --host-state-slots 2   # 64 GB edition
NINFER_DEVICES=0 ./serve-flash-next.sh /path/to/qwen3_8_flash_next_gsqrco_mtpB.ninfer                           # 64 GB edition, one GPU
./serve-tp2.sh /path/to/qwen3_8_27b_quasar_nvfp4.ninfer                           # Qwen3.8-27B
./serve-27b-16gb.sh /path/to/qwen3_8_27b_gsq3.ninfer                              # Qwen3.8-27B 3-bit, one GPU
NINFER_PORT=8099 ./serve-flash-next.sh /path/to/model.ninfer --api-key secret      # extra flags are appended
```
Requests name the model `capy-flash` (Flash-Next) or `qwen3.8-27b` (27B); `NINFER_MODEL_ID` or `--model-id` changes it:
```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"capy-flash","messages":[{"role":"user","content":"What is 17*23?"}]}'
```

Artifacts:
- Qwen3.8-Flash-Next: `Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash` (126.6 GB, NVIDIA's NVFP4 experts as published,
  SHA-256 in its `SHA256SUMS`), or convert `nvidia/Qwen3.8-Flash-Next-NVFP4` yourself with `tools/capy_flash/convert.py`
  from the source tree (same result).
- Qwen3.8-27B: `Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer` (17.4 GB, SHA-256 in its `SHA256SUMS`) or the official
  `Qwen3.8-27B-nvfp4-NInfer`; both run with the same flags. For artifacts without a vision component set `NINFER_VISION=0`.
- Qwen3.8-Flash-Next, 64 GB edition: `Feyd89/Qwen3.8-Flash-Next-GSQ-RCO-capy-flash` (102 GB).
- Qwen3.8-27B 3-bit for one GPU: `Feyd89/Qwen3.8-27B-GSQ-3bit-capy-flash` (12.5 GB; MMLU-Pro below the NVFP4 27B, see
  the main README).

`GET /health` answers 200 while the Engine accepts work and 503 after an Engine-wide failure; the
server then exits with status 2, so run it under a supervisor that restarts on a non-zero exit
(`Restart=on-failure`, or llama-swap, which reloads the model on the next request).
