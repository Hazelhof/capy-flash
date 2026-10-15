# capy-flash — prebuilt binaries

capy-flash (formerly ninfer-tp2) serves **Qwen3.8-Flash-Next** (MoE, NVFP4 experts, CPU offload) and **Qwen3.8-27B**
on two 16 GB Blackwell GeForce boards. Source and documentation: https://github.com/ValerioDolci/capy-flash.
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
| `SHA256SUMS` | digests of every file above; check with `sha256sum -c SHA256SUMS` |

## Requirements

- Linux x86_64 with the distribution named in the tarball (the binaries link its FFmpeg and libcurl:
  `libavcodec`, `libavformat`, `libavutil`, `libswscale`, `libcurl4`). On another distribution
  build from source or use the Dockerfile.
- An NVIDIA driver that supports CUDA 13.1 (R580 or newer). No CUDA toolkit is needed.
- Two `sm_120` boards with 16 GB for `--tp 2` (validated on 2x RTX 5070 Ti without P2P).
- **Qwen3.8-Flash-Next** also needs a CPU with AVX-512 (F/BW/VL, VNNI) and about **70 GiB of free RAM**: the routed
  experts that stay on the host are one 64.6 GiB page-locked allocation, and the server refuses to start without room
  for it (`NINFER_RAM_GUARD=0` skips the check).

## Run

```bash
tar -xzf capy-flash-*.tar.gz && cd capy-flash-*/
sha256sum -c SHA256SUMS
./serve-flash-next.sh /path/to/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer          # Qwen3.8-Flash-Next
./serve-tp2.sh /path/to/qwen3_8_27b_quasar_nvfp4.ninfer                           # Qwen3.8-27B
NINFER_PORT=8099 ./serve-flash-next.sh /path/to/model.ninfer --api-key secret      # extra flags are appended
```

Artifacts:
- Qwen3.8-Flash-Next: `Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash` (126.6 GB, NVIDIA's NVFP4 experts as published,
  SHA-256 in its `SHA256SUMS`), or convert `nvidia/Qwen3.8-Flash-Next-NVFP4` yourself with `tools/capy_flash/convert.py`
  from the source tree (same result).
- Qwen3.8-27B: `Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer` (17.4 GB, SHA-256 in its `SHA256SUMS`) or the official
  `Qwen3.8-27B-nvfp4-NInfer`; both run with the same flags. For artifacts without a vision component set `NINFER_VISION=0`.

`GET /health` answers 200 while the Engine accepts work and 503 after an Engine-wide failure; the
server then exits with status 2, so run it under a supervisor that restarts on a non-zero exit
(`Restart=on-failure`, or llama-swap, which reloads the model on the next request).
