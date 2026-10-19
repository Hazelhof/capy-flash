#!/bin/bash
# Live bench: server with --profile flash-next, prompts.jsonl (prompt, think, max_tokens, set, cat), fixed seeds;
# MTP acceptance and server-side decode from the request log (bench_an.py / bench_sum.py).
#   env: SERVE=<ninfer-serve>  BENCH_DIR=<dir with prompts.jsonl>  PY=<python>   usage: bench.sh <tag> <artifact .ninfer>
set -u
: "${SERVE:?}" "${BENCH_DIR:?}"; PY=${PY:-python3}
TAG=$1; ART=$2; O=$BENCH_DIR; L=$O/bench.log; PORT=${PORT:-18494}
CUDA_VISIBLE_DEVICES=0,1 $SERVE $ART --profile flash-next --port $PORT --model-id fn --request-log-jsonl $O/reqlog-$TAG.jsonl > $O/serve-$TAG.log 2>&1 & S=$!
for i in $(seq 1 300); do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null && break; kill -0 $S 2>/dev/null || break; sleep 1; done
echo "$(date +%T) $TAG start" >> $L
$PY - http://127.0.0.1:$PORT $O/prompts.jsonl $O/client-$TAG.jsonl <<'PY'
import json, sys, urllib.request
url, pf, out = sys.argv[1:]
with open(out, "w") as f:
    for i, l in enumerate(open(pf)):
        p = json.loads(l)
        body = {"model": "fn", "messages": [{"role": "user", "content": p["prompt"]}], "max_tokens": p["max_tokens"],
                "seed": 5000 + i, "chat_template_kwargs": {"enable_thinking": p["think"]}}
        r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                                                    {"Content-Type": "application/json"}), timeout=900))
        f.write(json.dumps({"set": p["set"], "cat": p["cat"], "think": p["think"], "i": i,
                            "tokens": r["usage"]["completion_tokens"]}) + "\n"); f.flush()
PY
kill $S; wait $S 2>/dev/null; for i in $(seq 1 60); do ss -ltn | grep -q ":$PORT " || break; sleep 2; done
echo "$(date +%T) $TAG done" >> $L
