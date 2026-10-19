# Bench analysis: per (set, category, reasoning) MTP acceptance, tokens per round, server-side decode.
import json
import sys
import collections
import os

O = os.environ.get("BENCH_DIR", "bench")
for tag in sys.argv[1:]:
    cl = [json.loads(l) for l in open(f"{O}/client-{tag}.jsonl")]
    rq = [json.loads(l) for l in open(f"{O}/reqlog-{tag}.jsonl") if '"request_done"' in l]
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0.0])
    for c, r in zip(cl, rq):
        sp = r["speculative"]
        for k in (
            (c["set"], c["cat"], c["think"]),
            ("ALL", c["cat"].split("-")[0] if c["cat"].startswith("code") else c["cat"], c["think"]),
        ):
            a = agg[k]
            a[0] += sp["accepted_tokens"]
            a[1] += sp["drafted_tokens"]
            a[2] += sp["rounds"]
            a[3] += r["result"]["completion_tokens"]
            a[4] += r["timings_seconds"]["decode"]
    for (st, cat, think), (acc, dr, rounds, toks, dec) in sorted(agg.items()):
        print(
            f"{tag:10s} {st:5s} {cat:8s} think={'on ' if think else 'off'} accept {acc / dr:.3f} tok/round {toks / rounds:.2f} decode {toks / dec:6.1f} tok/s ({toks} tok)"
        )
