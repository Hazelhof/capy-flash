<!-- Modified for fngine from llama.cpp/ggml (see below) @ 3cf03257f219afbe7334045ff7c6a06ac68c627d. See NOTICE. -->

# ggml-iq

Standalone extract of ggml's reference code for the quantization formats fngine reads.
It does not depend on `ggml.h`.

- Upstream: ggml (MIT), commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`. The copy used is the one vendored in
  Strata@30ec18ec7094550fcc594fd948220d511d80464e under `third_party/llama.cpp/ggml/`.
- License: `LICENSE` (ggml MIT license, copied verbatim).

## Extracted

| fngine file | upstream file | contents |
|---|---|---|
| `ggml-iq.h` | `ggml/src/ggml-common.h` | `QK_K`, `QK2_0`, `QK8_0`, `QK8_1`, `QK4_NL`, `IQ3S_N_SCALE`; block structs and size asserts for `block_q2_0`, `block_q8_0`, `block_q8_1`, `block_q8_K`, `block_iq2_s`, `block_iq3_xxs`, `block_iq3_s`, `block_iq4_nl`, `block_iq4_xs` |
| `ggml-iq.h` | `ggml/src/ggml-impl.h` | `fp32_from_bits`, `fp32_to_bits` (renamed `ggml_iq_*`), `ggml_compute_fp16_to_fp32`, `ggml_compute_fp32_to_fp16`, `ggml_compute_bf16_to_fp32`, `ggml_compute_fp32_to_bf16` |
| `ggml-iq-tables.h` | `ggml/src/ggml-common.h` | `kmask_iq2xs`, `ksigns_iq2xs`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`, `kvalues_iq4nl` |
| `ggml-iq.c` | `ggml/src/ggml-quants.c` | `best_index_int8`, `nearest_int`, `quantize_row_{q2_0,q8_0,q8_1,q8_K}_ref`, `dequantize_row_{q2_0,q8_0,q8_K,iq2_s,iq3_xxs,iq3_s,iq4_nl,iq4_xs}`, `quantize_row_iq4_nl_impl`, `quantize_iq4_nl` |
| `ggml-iq.c` | `ggml/src/ggml-cpu/quants.c` | `ggml_vec_dot_{q2_0,iq4_nl}_q8_0_generic`, `ggml_vec_dot_{iq2_s,iq3_xxs,iq3_s,iq4_xs}_q8_K_generic` |

## Changes

- `GGML_TABLE_BEGIN/END` replaced by `static const` arrays, so the tables header can be included from C, C++ and CUDA host code.
- `ggml_half`/`ggml_fp16_t` are `uint16_t`, `ggml_half2` is `uint32_t`; `block_q8_1` uses a plain anonymous union/struct instead of the `GGML_EXTENSION`/`GGML_COMMON_AGGR_*` macros; `static_assert` became `IQ_STATIC_ASSERT`.
- `GGML_RESTRICT` became `IQ_RESTRICT`; `GGML_CPU_FP16_TO_FP32` became `GGML_FP16_TO_FP32` (both are the same scalar conversion); `GGML_ASSERT` lines were removed. Plain `assert` calls are kept.
- Declarations are wrapped in `extern "C"`. Function bodies are otherwise verbatim.
