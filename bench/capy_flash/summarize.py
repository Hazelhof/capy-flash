#!/usr/bin/env python3
"""Summary of an agentic replay (replay.py output), optionally with the server-side decode rate.
Usage: summarize.py replay.jsonl [--ninfer-request-log reqlog.jsonl | --strata-engine-log engine.log]
Client side: TTFT median / p90, total time. Server side (the decode number to compare engines with: a
client that receives tokens in bursts overstates the rate): ninfer-serve --request-log-jsonl
(completion_tokens / decode seconds per request) or Strata's engine log ("N generated in X ms")."""
import json, re, statistics as st, sys
R = [json.loads(l) for l in open(sys.argv[1])]
ok = [r for r in R if not r["error"] and r["ttft_s"]]
tt = sorted(r["ttft_s"] for r in ok)
print(f"turns {len(R)}, ok {len(ok)} | TTFT median {st.median(tt):.2f} s, p90 {tt[int(0.9 * (len(tt) - 1))]:.2f} s | "
      f"total {sum(r['total_s'] for r in ok):.0f} s | cached share "
      f"{sum(r['cached_tokens'] or 0 for r in ok) / max(1, sum(r['prompt_tokens'] or 0 for r in ok)):.3f}")
if len(sys.argv) > 3 and sys.argv[2] == "--ninfer-request-log":
    v = [r["result"]["completion_tokens"] / r["timings_seconds"]["decode"] for r in map(json.loads, open(sys.argv[3]))
         if r.get("event") == "request_done" and r["timings_seconds"].get("decode", 0) > 0 and r["result"]["completion_tokens"] > 1]
    print(f"server-side decode: median {st.median(v):.1f} tok/s over {len(v)} requests")
elif len(sys.argv) > 3 and sys.argv[2] == "--strata-engine-log":
    v = [float(m.group(1)) for l in open(sys.argv[3], errors="replace") for m in [re.search(r"generated in [\d.]+ ms \(([\d.]+) tok/s\)", l)] if m]
    print(f"server-side decode: median {st.median(v):.1f} tok/s over {len(v)} requests")
