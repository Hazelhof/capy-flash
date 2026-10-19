# ABCCBA bench summed per arm (both runs) and per set (held / t186 / all): server-side decode.
import json
import collections
import os

O = os.environ.get("BENCH_DIR", "bench")
ARMS = {"old": ["old1", "old2"], "base": ["base1", "base2"], "refB": ["refB1", "refB2"]}


def load(tag):
    cl = [json.loads(l) for l in open(f"{O}/client-{tag}.jsonl")]
    rq = [json.loads(l) for l in open(f"{O}/reqlog-{tag}.jsonl") if '"request_done"' in l]
    return zip(cl, rq)


agg = collections.defaultdict(lambda: [0, 0, 0, 0.0, []])
for arm, tags in ARMS.items():
    for tag in tags:
        run = collections.defaultdict(lambda: [0, 0.0])
        for c, r in load(tag):
            lang = "code" if c["cat"].startswith("code") else c["cat"]
            for st in (c["set"], "all"):
                k = (arm, st, lang, c["think"])
                a = agg[k]
                sp = r["speculative"]
                a[0] += sp["accepted_tokens"]
                a[1] += sp["drafted_tokens"]
                a[2] += r["result"]["completion_tokens"]
                a[3] += r["timings_seconds"]["decode"]
                run[(st, lang, c["think"])][0] += r["result"]["completion_tokens"]
                run[(st, lang, c["think"])][1] += r["timings_seconds"]["decode"]
        for k, (t, d) in run.items():
            agg[(arm,) + k][4].append(t / d)
print(
    f"{'set':5s} {'lang':5s} {'think':3s} | {'old acc':>7s} {'tok/s':>12s} | {'base acc':>8s} {'tok/s':>12s} | {'refB acc':>8s} {'tok/s':>12s} | refB/old  refB/base"
)
for st in ("all", "held", "t186"):
    for lang in ("it", "en", "code"):
        for th in (False, True):
            row = {arm: agg[(arm, st, lang, th)] for arm in ARMS}
            if not row["old"][1]:
                continue
            f = lambda a: f"{a[0] / a[1]:7.3f} {a[2] / a[3]:6.1f} ({'/'.join(f'{x:.0f}' for x in a[4])})"
            sp = {k: v[2] / v[3] for k, v in row.items()}
            print(
                f"{st:5s} {lang:5s} {'on' if th else 'off':3s} | {f(row['old'])} | {f(row['base'])} | {f(row['refB'])} | {sp['refB'] / sp['old'] - 1:+7.1%} {sp['refB'] / sp['base'] - 1:+7.1%}"
            )
