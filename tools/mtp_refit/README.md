# MTP draft refit (Qwen3.8-Flash-Next)

Re-distills the dense part of Qwen3.8-Flash-Next's MTP layer on the output of *this* engine (NVFP4 experts, the
served numerics), so the draft accepts more tokens per round. The verifier is unchanged and verification is exact, so
the generated tokens are the same as without drafts; only speed changes.

## Result (measured, 2× RTX 5070 Ti, `--profile flash-next`, server-side decode, 280 requests per arm, ABCCBA)

| language | reasoning | v0.5.4 (released) | + Italian draft shortlist | **+ refit head** | vs v0.5.4 |
|---|---|---|---|---|---|
| Italian | off | 107.8 tok/s | 115.3 | **127.6** | **+18.4 %** |
| Italian | on | 106.5 | 114.6 | 116.9 | +9.7 % |
| English | off | 122.8 | 121.2 | 128.3 | +4.6 % |
| English | on | 118.3 | 116.7 | 122.8 | +3.8 % |
| code | off | 135.5 | 136.8 | 146.4 | +8.0 % |
| code | on | 109.2 | 108.3 | 112.1 | +2.7 % |

Held-out prompts only (never in training) plus 30 prompts from a different source; no category regresses. Identity
check (greedy, frozen expert residency): MTP with drafts = MTP without drafts, 20/20 sequences.

## Recipe
1. **Prompts** (`build_prompts.py`): 2,350 public prompts — Italian (alpaca-gpt4-italian, evol-instruct-italian, incl.
   ~300 coding requests in Italian), English (UltraChat test_gen), code (Magicoder OSS-Instruct); 30 % with reasoning
   on; 40 + 15 held-out per category.
2. **On-policy data** (`gen_dump.sh`): the engine answers them (TP2, MTP 3, int8 KV, T 1 / top-k 20 / top-p 0.95), then
   a teacher-forced pass dumps, per position, the wide residual after layer 47 (the MTP input, bf16, 20 KB) and the
   target's top-20 log-probs (`NINFER_QWEN4_TFDUMP_COMPACT=1 NINFER_QWEN4_TFDUMP_TOPK=20`, `score-windows`;
   reader: `compact.py`). ~2.3 M answer tokens, ~60 GB.
3. **Training** (`train.py`, `mtp_model.py`, `data.py`): the 88.9 M dense parameters (stem, attention, hyper-connections,
   router, shared expert); routed experts, embedding, head and indexer frozen. Soft cross-entropy to the target top-20
   (restricted to the draft shortlist) at each of 3 unrolled depths, teacher-forced as served (draft steps attend the
   depth-0 keys only). One window per whole sequence: a window that does not start at position 0 loses the attention
   sink and the loss explodes. AdamW lr 1e-4, 2 epochs, 8,192 anchors per step: ~1 h on one 16 GB GPU.
4. **Artifact** (`make_artifact.sh`): a shadow checkpoint (symlinks to the NVIDIA checkpoint + the refit tensors, index
   pointing at them) through `tools/capy_flash/convert.py`.
5. **Bench** (`bench.sh`, `bench_an.py`, `bench_sum.py`, `eval_table.py`).

Notes: three recipes (1 epoch lr 3e-5; 2 epochs lr 1e-4; depth weights 0.5/1/1.5) give the same held-out gain — depth 0
improves, depths 1-2 do not move: the limit is the layer with frozen experts, not data or optimization. Before
starting Flash-Next with its ~65 GB pinned arena after large reads, drop the clean page cache (`evict_all.py <dirs>`),
otherwise systemd-oomd may kill the server during the allocation.

The PyTorch MTP layer follows `tools/mtp_refit/mtp_ref.py` of
[ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2)
(Apache-2.0), adapted to the NVIDIA checkpoint (FP8 block-128 MTP experts) and validated against this engine (live
acceptance within 0-2 points of the reference).

Environment: `MTP_REFIT_CHECKPOINT` (NVIDIA ModelOpt checkpoint dir), `MTP_REFIT_SHORTLIST` (draft shortlist file),
`CORPUS`, `OUT`, `DRV`, `ART`, `W`, `SERVE`, `BENCH_DIR`; Python with torch ≥ 2.14 and transformers ≥ 5.16 (`qwen4_exp`).
