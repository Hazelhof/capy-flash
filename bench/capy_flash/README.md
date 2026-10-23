# Agentic replay (capy-flash benchmark)

The replay behind the speed table of the main README: 3 synthetic coding-agent sessions (`sessions.json`, 53 turns,
system prompt 8.6k tokens, prompts growing to ~81k tokens, ~3.8k new tokens per turn), each turn sent as a full
chat request with the whole history, streamed, temperature 0, 256 output tokens. A server with a prefix cache
re-reads only the new part of each turn. It measures what an agent feels: time to first token, decode speed and
total time.

```
# 1. start the server under test (capy-flash below, see the main README for the build and the artifact)
CUDA_VISIBLE_DEVICES=0,1 build/apps/ninfer-serve <artifact>.ninfer --profile flash-next --host 127.0.0.1 \
  --port 8098 --model-id capy-flash --greedy --request-log-jsonl reqlog.jsonl
#    (one GPU, 64 GB edition: --profile flash-next-1gpu; the profiles are listed in docs/serving.md#profiles)
# 2. replay, then summarize
python3 bench/capy_flash/replay.py http://127.0.0.1:8098 capy-flash bench/capy_flash/sessions.json out.jsonl 256
python3 bench/capy_flash/summarize.py out.jsonl --ninfer-request-log reqlog.jsonl
```

Compare engines with the **server-side** decode rate: a server that streams tokens in bursts makes the client-side
rate look higher than it is. For Strata, start its OpenAI server (`python -m serve.server --engine strata --config
<config> --port <port>`) with the same model id in the replay and pass its engine log to `summarize.py
--strata-engine-log`. The published Strata 0.1.44.1 runs used two GPUs (layer split auto, `--pipeline-windows 2`,
`--pcie-frac 0.2` for IQ3_S, its NVFP4 default otherwise), `--max-context 131072`, int8 KV, MTP (`--spec 4
--spec-min-p 0.5`), greedy; for the NVFP4 pack `STRATA_NO_ARENA_THP=1` (without it the 63 GiB expert arena is
requested with transparent huge pages and, on a machine with THP defrag `madvise`, compaction stalls the load).

`gen_sessions.py` documents how `sessions.json` was built (from a source tree and the model tokenizer); the published
numbers all used the `sessions.json` committed here.
