# Qwen4Exp model reference

This reference defines the mathematics of `Qwen4ExpForCausalLM` (`qwen4_exp_text`), the
architecture of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(`Qwen4ExpForConditionalGeneration`). It is **not** an instance of `Qwen3_5MoeForCausalLM`:
hyper-connection residual streams, Qwen Sparse Attention (QSA) and the n-gram PLE embedding
change the block and state mathematics.

**Status in this tree:** the configuration is interpreted (`models/qwen3_5/config.{h,cpp}`,
`Architecture::Qwen4Exp`), the ModelOpt NVFP4 checkpoint converts (`tools/convert/qwen4_exp.py`,
recipes `qwen3_8_flash_next_nvfp4` and `qwen3_8_flash_next_nvfp4_dense`) and the artifact loads and
binds (`load/text.cpp` `bind_qwen4_text`/`bind_qwen4_mtp`); see [Artifact and loading](#artifact-and-loading).
Text executes at one device (`--tp 1`) and at two (`--tp 2`): chunked Prefill, CausalScoring and
ordinary decode with CUDA Graphs, and MTP speculative decoding (`--spec mtp --draft-tokens K`,
K = 1..5, full-vocabulary proposal; at tp 2 with the offload expert backend only); see
[Execution](#execution). Vision, DFlash and a trimmed proposal head are refused. This reference
and its artifact
naming come from giveen/ninfer-ext (Apache-2.0), which executes the architecture.

The upstream authorities are:
- Transformers `models/qwen4_exp` for Text;
- SGLang `qwen4_exp_mtp.py` for MTP, because Transformers ignores `mtp.*`.

Shared pieces keep their [Qwen3.5 reference](qwen3_5-model.md) definitions:
- GDN recurrence;
- gated GQA;
- interleaved MRoPE;
- Vision tower and multimodal positions;
- tokenizer and frontend semantics.

This document lists only what differs.

## Configuration

| Quantity | Qwen3.8-Flash-Next |
|---|---:|
| Hidden width H / hyper-connection streams hc / wide width hc·H | 2560 / 4 / 10240 |
| Text layers; GDN / QSA attention | 48; 36 / 12 (every fourth layer is attention) |
| Q heads / KV heads / head dimension; rotary dimensions | 24 / 2 / 256; 64 |
| QSA indexer heads / key heads / head dimension; budget / block | 4 / 1 / 128; 2048 / 4 |
| GDN key / value heads, head dimensions, conv width | 16 / 48, 128 / 128, 4 |
| Routed experts / selected / expert width; shared width | 512 / 10 / 640; 640 |
| HC low rank | 320 |
| PLE layer (one-indexed `ple_layer_ids`) | 2, i.e. block index 1 |
| N-gram size / heads per n-gram / row width / table rows | 3 / 8 / 160 / 320,001,536 |
| Vocabulary; position capacity | 248,320; 262,144 |

Other fixed values:
- `rms_norm_eps=1e-6`
- `rope_theta=1e7`
- `mrope_section=[11,11,10]`
- `output_gate_type=sigmoid`
- `norm_topk_prob=true`
- `seed=1234`
- `eos_token_id=248044`

Embedding and output head are untied, and there is no final RMSNorm: the output mixer feeds the
head directly.

The BF16 checkpoint holds about 180B parameters, of which about 121B are routed experts and 51.2B
are the n-gram table.

## Norms

- `offset_rmsnorm(x, w) = x / rms(x) * (1 + w)`. It is used by attention Q/K norms, the indexer
  norms, HC norms, PLE norms and the MTP pre-fc norms.
- A **grouped** offset RMSNorm computes the RMS separately over each contiguous H-wide stream of a
  wide vector and then applies the hc·H-wide weight.
- The GDN output norm is `w * x / rms(x) * act(z)` with a plain (not offset) weight, and
  `act = sigmoid` here. Qwen3.5 uses SiLU in this position.

## Hyper-connection residual

The residual is a wide vector `R` of hc streams, each H wide. Embedding initializes every stream
with the token embedding (`R = [e, e, e, e]`).

Each block has two HC mixers, one around the mixer and one around the MoE. Each mixer computes:

```text
n      = grouped_offset_rmsnorm(R, hc_norm)                         # [hc·H]
m      = sigmoid(W_up · silu(W_down · n / hc))                      # [hc·H], W_down: [320, hc·H]
x      = mean_s(m_s ⊙ n_s)                                          # block input, [H]
inject = 2 · sigmoid(W_inject · n / hc)                             # [hc]
y      = block(x)                                                   # [H]
R      = R + [inject_s · y for each stream s]
```

The final `hyper_connection_mixer` has no inject weight. Its `x` is the Text hidden state that
goes to `lm_head`.

HC is per token and has no recurrent state. The wide residual is the continuation hidden that MTP
consumes.

## Block

```text
if block has PLE:  R = R + PLE(R, tokens)
x, inj = HC_attn(R);  R = R + inj ⊗ mixer(x)        # GDN or QSA gated attention
x, inj = HC_mlp(R);   R = R + inj ⊗ SparseMoE(x)
```

**GDN** is the Qwen3.5 formulation:
- `in_proj_qkv` feeds a causal depthwise conv (kernel 4) followed by SiLU;
- Q and K are L2-normalized, and Q is additionally scaled by `1/sqrt(128)`;
- each key head serves three consecutive value heads (`repeat_interleave`);
- `β = sigmoid(b)` and `g = -exp(A_log) · softplus(a + dt_bias)`;
- the delta-rule recurrence runs over `[Dk,Dv]` state per value head;
- the output passes through the gated norm with the `z` gate, then `out_proj`.

**Sparse MoE** is the [Qwen3.5 formula](qwen3_5-model.md#sparse-moe) with E = 512 and K = 10:
- softmax over router logits, top-10, and renormalization of the selected weights;
- SwiGLU experts whose fused `gate_up_proj` rows are gate first, then up;
- the shared expert scaled by `sigmoid(shared_expert_gate · x)`.

## Qwen Sparse Attention

The attention projections are unchanged from Qwen3.5:
- `q_proj` emits `[q_h, gate_h]` per head;
- offset-RMSNorm is applied to Q and K;
- partial interleaved MRoPE covers the first 64 dimensions;
- the output is `o_proj((softmax(qKᵀ/16)V) ⊙ sigmoid(gate))`.

QSA restricts which cached keys each query may see.

The indexer computes `index_qk_proj(x)`, giving `[4·128 | 128]`: the query heads and one raw key.
- **Raw key:** cached per token (a new KV-like plane), with no norm and no RoPE.
- **Query heads:** offset-RMSNorm, then RoPE on their first 64 dimensions at the query position.

For a query that sees cache indices `[0, n)`:

```text
blocks  = floor(n / 4); block b covers indices [4b, 4b+4)
k̄_b    = RoPE(offset_rmsnorm(mean(raw_key[4b .. 4b+3]), k_layernorm), position(4b))
score_b = Σ_heads relu(q_h · k̄_b) / sqrt(128)
keep    = min(512, blocks) blocks with the largest scores
visible = tokens of kept blocks ∪ tail [4·blocks, n)
```

Selection width is at most `2048 + 3` tokens. While `n ≤ 2051`, every token is visible and QSA
equals dense causal attention; beyond that, decode attention cost stays constant.

- **Positions:** pooled keys use the three-axis position of the block's first token.
- **Ties:** the oracle breaks exact score ties toward the lower block index. Production selection
  is qualified against the oracle with an explicit near-tie allowance.
- **Dtypes:** upstream stores raw keys and pooled means in the activation dtype. That is a private
  staging choice, not a semantic boundary.

## PLE n-gram embedding

PLE runs once, before block 1's attention mixer.

**N-gram rows.** For the token at position p and n-gram size n ∈ {2, 3}, the context token at
distance s is:
- `t[p-s]` when none of `t[p-s .. p-1]` is EOS;
- EOS otherwise.

An EOS therefore restarts every window, exactly like sequence start, where the two-token history
is EOS-filled. Head j of n-gram n uses row:

```text
mixed_n = XOR_{k<n} (ctx_k · m_k)            # signed 64-bit; products stay below 2^63
row_j   = mixed_n mod P_j + offset_j
```

The rows use these constants:
- **Multipliers** `m = [23703573157769, 20109073645365, 8052911324071]`. They are derived from
  the seed by SplitMix64 and also stored as the `layer_multipliers` buffer.
- **Moduli** `P_j`: the 16 consecutive primes after `ngram_vocab_size_base − 1`, from 20,000,003
  to 20,000,171.
- **Offsets:** their running sum.

The table has `320,001,536 × 160` rows, split in the checkpoint into 128 shards of 2,500,012 rows.
The 16 gathered rows are concatenated, 2-gram heads first, into the 2560-wide embedding `E`.

**Injection.** With `R` the wide residual entering the block:

```text
k_s  = grouped_offset_rmsnorm(key_proj · E, norm_key)_s            # key_proj: [hc·H, 2560]
v    = value_proj · E                                              # [H]
q_s  = grouped_offset_rmsnorm(R, norm_query)_s
g_s  = (k_s · q_s) / sqrt(H);  g_s = sign(g_s) · sqrt(max(|g_s|, 1e-6))
u    = [sigmoid(g_s) · v for each s]                               # [hc·H]
c    = grouped_offset_rmsnorm(u, norm_conv)
R    = R + u + silu(depthwise_conv(c; kernel 4, dilation 3))
```

**State.** The convolution is causal with dilation 3, so its state is the previous nine `c`
vectors (9 × 10240). Together with the two-token n-gram history, this is new per-sequence
recurrent state beside the GDN state.

## MTP

The single MTP layer is a full QSA block with its own KV and indexer-key planes and no PLE. It
consumes the **wide** residual `R_t`, not a mixed or normalized hidden state. The token-shift
alignment is the [Qwen3.5 one](qwen3_5-model.md#prefill-decode-and-mtp): hidden states and
positions stay in place while tokens shift by one.

```text
e   = fc_embedding · offset_rmsnorm(embed(x_(t+1)), pre_fc_norm_embedding)       # [H]
h   = offset_rmsnorm(R_t, pre_fc_norm_hidden)     # one RMS over all hc·H values, not grouped
R'  = [fc_hidden · h_s + e for each stream s]
R'  = QSA_block(R')                                # HC mixers, QSA attention, 512×top-10 MoE
logits = lm_head(mtp.hyper_connection_mixer(R'))
```

`R'` is the draft continuation hidden for recursive proposals. The draft MoE uses its own expert
banks (`mtp.layers.0.mlp.*`).

## Vision

The tower has the Qwen3.5 geometry (depth 27, width 1152, patch 16, merge 2). Its merger emits
2560-wide columns, which replace placeholder embeddings before the embedding is repeated into the
hc streams. PLE hashes the placeholder token IDs themselves.

## Per-sequence state

Per sequence, State holds:
- GDN conv and recurrent state (as Qwen3.5);
- PLE conv history (9 × 10240) and n-gram token history (2);
- attention K/V plus one raw indexer key per token for each of the 12 attention layers and the
  MTP layer;
- the wide continuation hidden (10240).

## Artifact and loading

Logical parameters (converter `tools/convert/qwen4_exp.py`, binder `load/text.cpp`; names follow
ninfer-ext, so its `qwen3_8_flash_next_nvfp4` artifact binds here as well):

| Logical parameter | Source | Residency |
|---|---|---|
| `text/token_embedding`, `text/output_head` | `embed_tokens`, `lm_head` | device (embedding HostMapped with `--embedding-host`) |
| `text/hc_head/{norm,down,up}` | `hyper_connection_mixer.*` | device |
| `text/layers/<i>/{attn_hc,ffn_hc}/{norm,down,up,inject}` | `{attn,mlp}_hyper_connection.*` | device |
| `text/layers/<i>/attention/*` (+ `indexer_{query,key,query_norm,key_norm}`) | `self_attn.*` | device |
| `text/layers/<i>/gdn/*` | `linear_attn.*` | device |
| `text/layers/<i>/moe/{router,shared_score,shared/*}` | `mlp.{gate,shared_expert_gate,shared_expert}` | device |
| `text/layers/<i>/moe/experts/<e>/{gate,up,down}` | `mlp.experts.<e>.*_proj` | **HostMapped** stacked banks |
| `text/layers/<ple>/ple/table` | `ple.ple_embedding.ngram_embedding.shard_*` + `weight_scale` | **HostFile** (read-only mapping) |
| `text/layers/<ple>/ple/{key,value,*_norm,convolution}` | `ple.*` | device |
| `mtp/{embedding_norm,hidden_norm,embedding_projection,hidden_projection}`, `mtp/hc_head/*`, `mtp/layers/0/*` | `mtp.*` | as the Text roles |

Routed experts: each layer's gate/up projections (expert-major: gate then up of expert 0, then
expert 1, ...) form one NVFP4 parent `[E*2*I, H]` and the down projections one parent `[E*H, I]`.
Both are stacked planes (tensor object `divisors` = `2E` and `E`): every source matrix keeps its own
ModelOpt divisor, and an expert is a fixed row range of each bank, i.e. three contiguous byte spans
per bank (codes, swizzled scales, divisor words; `models::qwen3_5::routed_expert_spans`). One
Flash-Next expert is 2,764,800 bytes of codes and scales plus 12 divisor bytes. The Text experts
keep the checkpoint's codes, scales, weight divisors and per-matrix activation divisors
(`import_encoded`, AllowA4); the MTP experts (block FP8 in the checkpoint) are re-encoded by
`nvfp4_mse_stacked` with one divisor per matrix (A16Only). The Qwen4Exp shard rule places every
bank `ShardAxis::HostMapped`: one page-locked allocation (2 MiB pages when available) that every
device reads over PCIe and an expert cache copies from; 48 x 512 + 512 experts take 64.6 GiB.

At `--tp 2` the dense parameters split by heads, the routed experts stay one host-mapped copy
partitioned by expert ID at run time, the hyper-connections, router, indexer and PLE are
replicated (or the mixers split by residual streams), and the shared expert alternates between
the ranks by block: [Two-GPU tensor parallelism §13](tensor-parallel.md#13-qwen4exp-qwen38-flash-next-at-tp-2).

The n-gram table stays in the artifact file: `Residency::HostFile` maps its object ranges
read-only (`MADV_RANDOM`, crossing `.part` files as several segments); host code gathers rows from
`Model::file_weight(...).mapped`. Its format is `fp8_e4m3fn_row_bf16` with the checkpoint's single
BF16 multiplier repeated on every row.

Dense projections: the `qwen3_8_flash_next_nvfp4` recipe stores them Q8 with a Q6 head (ninfer-ext's
choice); `qwen3_8_flash_next_nvfp4_dense` stores the attention and GDN projections, the shared
expert and the output head as NVFP4 (`nvfp4_mse`, A16Only) and keeps hyper-connections, the
indexer, the PLE projections and the MTP stem Q8. Routers, shared-expert gates, norms and vectors
stay direct.

## Execution

`execution/qwen4_forward.h` (one template for execution and startup workspace sizing, at one
device or over the two ranks of `--tp 2`), entry points in `execution/qwen4_text.cpp`, Program
state in `program/` and `state/`:

- **Residual**: `hyper_connection_expand` of the embedding; every mixer is
  `hyper_connection_combine_mix` (the pending block output is committed inside the next mixer); the
  output mixer (`hyper_connection_mix`, no inject) feeds the head. The carried hidden of every
  round, Prefill column, StateImage continuation and CausalScoring tile is the committed **wide
  residual** [S*H]; the output mixer runs where logits are needed.
- **QSA**: `qsa_append` (K/V plus the index records and pooled block keys), `qsa_select`,
  `qsa_attention` for every call, also below the dense-equivalent extent (the selection is then all
  visible keys). The two index planes of every attention layer live in the Main KV page pool after
  the K/V planes (`DecoderStateSpec::qsa_index`), so paging, prefix reuse and host tiers move them
  with the K/V bytes.
- **GDN**: Linear projection, `causal_conv1d_silu_split` (Prefill) or `gdn_projected_conv_snapshot`
  (decode), `gated_delta_net[_batch_update]`, `gated_rmsnorm_sigmoid`.
- **PLE**: rows gathered on the host (`PleGather`, mapped table with prefetch): per Prefill chunk
  through pinned staging, per decode round into a pinned buffer the round uploads. The convolution
  history (9 x 10240 BF16 per slot) is StateImage state (`PleStateSpec`).
- **MoE**: `moe_route` (top-10 renormalized, sigmoid shared gate), the shared expert as Q8 Linear,
  the routed experts through a backend that returns the FP32 partial sum, and `moe_combine_shared`
  (one BF16 rounding). Backends (`execution/qwen4_experts.h`, `EngineOptions::routed_experts`):
  - **Offload** (default): a device slot cache per layer (T14 `ExpertCache`, slot table read by
    `moe_slots_nvfp4`, W4A4) for the hot experts, and the CPU for the routed experts the table does
    not hold (`ops/sparse_moe/host`: `HostMoeMailbox` + `HostMoeExecutor`, NVFP4 W4A8 AVX-512, 6
    pinned threads). Per layer, inside the round's CUDA Graph: right after the router a publish
    kernel and two stream memory writes hand x, the routing and the host-side ids to the CPU; the GPU
    runs the shared expert and the resident experts; a stream memory wait (no SM held) and an
    accumulate kernel add the CPU's FP32 partial; `moe_combine_shared` rounds once. A prefill chunk
    is one handshake per layer for all its columns (the CPU reads each missing expert once per
    chunk, in dot groups of <= 8 tokens), the resident experts in calls of <= 8 columns. Optionally
    (`device_reads`, decode calls) the first N missing experts are read by the GPU itself from the
    host bank over PCIe on a second stream while the CPU computes the rest.
  - Cache policy on the host, overlapped with the device: every execution is one window
    (`begin_window`/`observe_window`/`end_window`, decode round or prefill chunk). The publish
    kernel also logs each layer's full routing to mapped memory; during a decode round the execution
    thread reads it layer by layer (LRU, optional decayed-frequency admission, insertion cap per
    layer per window) and issues the copies (`ExpertCache::submit_fills_unfenced`) as soon as the GPU
    has published the next layer, i.e. has left the layer whose slots are rewritten; a copied expert
    is published at the first window that begins after its copy. A prefill chunk is a prompt
    window: its most routed experts per layer become resident (`decide_prompt`, no cap).
  - Slots per layer: `routed_experts.slots_per_layer`, or 0 = the device memory the rest of the
    Program leaves minus `slot_headroom_bytes` (Engine resolves it before the final plan;
    `MemorySummary::routed_expert_slots`).
  - Elastic KV (T64, `EngineOptions::elastic_kv`, driver `--elastic [--kv-grow T]
    [--kv-budget-gib G]`): instead of a fixed partition sized for `max_context`, the KV pools keep
    max_context's addresses but are backed page by page, and the expert cache takes every other
    byte. Each rank's persistent arena is a `core/elastic_arena.h` VMM reservation (2 MiB granules)
    whose elastic ranges are the KV page planes (page-major) and the tail of each expert plane. The
    slots are `slots_per_layer` base slots (sized for max_context's KV, or `kv_budget_bytes`;
    layer-major, lent to prompts as before) plus `routed_experts.elastic_columns` columns of one
    slot per layer, numbered column-major after the base and prefetch staging slots so the last
    columns are one contiguous tail per plane (`ExpertCachePolicy::pool_slot`). A KV pool that
    hands out a page past its backing calls `program/storage/qwen4_elastic.h` first: the backing
    rises to the next multiple of `grow_tokens`, and while the arena would exceed its startup bytes
    the cache gives up its last column (`ExpertCache::set_active_columns`: fills drained, per layer
    the coldest experts dropped - lowest decayed frequency, then LRU - and the survivors of the
    column copied device-to-device into the freed kept slots, tables applied, device idle) before
    its granules are released and the KV granules backed. Before each prompt the backing is
    shrunk to max(live pages, prompt + one step) and the freed bytes come back as empty columns
    (refilled by later insertions). Between windows only (the KV materialization points); one
    request at a time. A request whose KV does not fit (or exceeds `kv_budget_bytes`) fails before
    any page is handed out.
  - **Mapped**: every expert read in place from its host-mapped bank by `moe_slots_nvfp4` (slot
    table = identity), W4A4; the M1 reference route.
  W4A4 uses one activation divisor per layer and projection input (the smallest of the layer's
  experts); the CPU path quantizes activations to int8 per 16 values (W4A8), so an expert's result
  depends on where it runs (closer to the BF16 reference on the CPU).
- **Measurement aids** (off by default): `NINFER_QWEN4_EXPERT_TRACE=<file>` (one JSON line per
  window: hits, CPU experts and time, fills, device stamps) and `NINFER_QWEN4_ROUTING_DUMP=<file>`
  (every observed layer's routing, binary).
- **Two devices** (`--tp 2`, placement in
  [tensor-parallel §13](tensor-parallel.md#13-qwen4exp-qwen38-flash-next-at-tp-2)): both ranks run
  the same schedule in lockstep on their own copy of the wide residual (hyper-connection mixers,
  router, indexer and PLE replicated, deterministic kernels, so both ranks route every token alike);
  each rank runs its QSA heads (12 query + 1 KV, its own KV and index planes) and GDN heads (8 key
  + 24 value, its own state), the routed experts of its ID parity (its own backend instance; with the
  offload backend one expert cache spans both GPUs, partitioned by ID parity: each rank runs the
  experts resident in its own slots, rank 0 publishes to the CPU every routed expert resident on
  neither GPU, reading a residency mirror of both slot tables kept on GPU 0, and the CPU's FP32
  partial joins rank 0's partial before the all-reduce; rank 0's window hooks drive both tables) and, on its blocks (`block % 2`), the
  shared expert, added into its FP32 partial (`moe_accumulate_shared`). Two exchanges per block:
  the mixer output (row-parallel projection, BF16 all-reduce) and the MoE (`allreduce_sum_f32`,
  rounded to BF16 once). The output head is split by vocabulary and gathered to rank 0, which
  samples. Rank 1's PLE rows come from rank 0's host gather (uploaded to both), its PLE
  convolution states from its StateImage mirror. The mailbox slot carries `hidden x (K+1)` FP32.
  The stream-split hyper-connection mixers are refused at tp 2. MTP at tp 2 runs the verification,
  the MTP layer (stem replicated, its QSA heads and expert partition per rank, shared expert on
  rank 0) and the proposal (replicated MTP output mixer, vocabulary-split head) over both ranks;
  rank 1 keeps its own PLE record and folds it with the same commit rows. The prefix-reuse bridge
  (`qwen4_mtp_forward_batch_tp2`) runs the MTP column of the reuse frontier and its proposal steps
  the same way, each rank from its own retained copy of the target hidden: rank 1 keeps its prompt
  chunk's wide residual in its own `prefill_hidden` and writes its capture checkpoints (T66b).
- **MTP** (`Qwen4Forward::run_mtp`, round schedule `program/speculative/mtp.cpp`): the stem over
  the target's wide residual and the shifted token, one QSA block over the MTP KV cache (layer 0,
  its own index planes) with the MTP experts, the MTP output mixer and the Text output head. Prompt
  chunks append every column's K/V and index records (stem + attention mixer + `qsa_append` only);
  the last prompt column runs the layer and drafts K tokens autoregressively. A round verifies the
  window `[anchor, d1..dK]` (K+1 columns per lane) through the Text forward in record mode: GDN
  writes ReplaySSM records (`gdn_projected_conv_record`, `gated_delta_net_replay_record`), PLE
  copies its convolution inputs to `Qwen4Runtime::ple_record`, QSA appends K/V and index records in
  place. Acceptance is tp2's (`speculative_accept_greedy_drafts`: greedy prefix match, or rejection
  sampling against the one-hot draft at temperature > 0). The commit folds the accepted prefix:
  `gdn_replay_fold` over the 36 GDN layers and `ple_conv_advance` of the recorded PLE inputs
  (`ProgramImpl::fold_qwen4_ple`); QSA needs no rollback: rejected positions are re-appended in
  position order by the next round, which re-publishes any pooled block key they had completed. The
  PLE n-gram rows of the window's columns are gathered on the host before the launch (the drafts
  are known from the previous round's egress). The MTP experts run through the Text backend as
  one more layer (`num_hidden_layers`): with the offload backend they have their own cache layer
  (slots, policy) and their misses go to the CPU like the Text layers' (the round's MTP calls,
  alignment and drafts, publish the layer once each; the policy observes the MTP layer's whole
  routing log at the end of the window); with the mapped backend they are read in place. Every
  execution that reaches the experts is one window: decode and MTP rounds, prefill chunks with
  their MTP prompt calls, and the prefix-reuse MTP bridge (`ProgramImpl::in_qwen4_window`). The
  checkpoint calibrates no activation scale for them (recipes store them A16Only): the
  W4A4 expert Op takes the widest calibrated range of the 48 Text layers times 32
  (`kQwen4MtpActivationRangeMargin`: inside the range where the W4A4 error is flat for both
  inputs; measurement T24 in `docs/capy-flash/development-log.md`). A bank with one
  weight divisor (ninfer-ext's artifact) is expanded into per-expert planes by the Program.
- **Validation dumps**: `NINFER_QWEN4_TFDUMP=<dir>` writes tfdump/1 files (format in
  `src/models/qwen3_5/execution/qwen4_tfdump.h`): logits per scored or decoded position, and the wide residual after each layer in
  eager Prefill; with `NINFER_QWEN4_TFDUMP_MTP=1` (eager, MTP) every prompt column runs the MTP
  layer and its draft logits go to `mtp_logits/`. `NINFER_QWEN4_TOP2=<file>` appends the target's
  top-2 logits of every decoded or committed verification column; `NINFER_QWEN4_MTP_PROBE=<file>`
  (eager) the inputs of every MTP-layer expert call.

- **Validation knobs** (environment, read when the Program is constructed; none is a product
  option). Observation only: `NINFER_QWEN4_EXPERT_TRACE`, `NINFER_QWEN4_ROUTING_DUMP`,
  `NINFER_QWEN4_TFDUMP` (`_LAYERS=<list>|all|none`, `_LOGITS=0`, `_MTP=1` above),
  `NINFER_QWEN4_TOP2`, `NINFER_QWEN4_MTP_PROBE`. Execution changes that keep the tokens (greedy):
  `NINFER_QWEN4_MTP_NO_DRAFTS=1` (every MTP round verifies the anchor alone, extent 0: the
  no-draft control of an MTP run) and `NINFER_QWEN4_MTP_ORACLE=<file>` (drafts taken from the
  matching line of whole sequences, prompt included: fully accepted rounds). Token-changing, refused
  unless `NINFER_QWEN4_VALIDATION=1` is set too (`execution::require_qwen4_validation_opt_in`):
  `NINFER_QWEN4_FORCE_TOKENS=<file>` (the committed token of a decode step is the matching line's
  next token; single-token commits only, so it throws once an MTP round accepts a draft: use it
  with `NINFER_QWEN4_MTP_NO_DRAFTS=1`), `NINFER_QWEN4_TFDUMP_FORCE=<ids file>` (teacher-forced
  decode, prompt included) and `NINFER_QWEN4_EXPERT_FREEZE=1` (offload backend: residency decided
  by the prompt windows alone, each prompt window waits for its fills, decode windows insert
  nothing, so an MTP run and its no-draft control run every expert on the same side and produce
  identical tokens). Under MTP the forcing knobs rewrite only the host token; the device built the
  round's MTP column from the unforced one.
