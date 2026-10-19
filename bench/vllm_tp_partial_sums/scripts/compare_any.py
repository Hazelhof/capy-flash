"""Distances of several runs from one reference (first argument): NLL/perplexity with paired CI, KL, top-1."""

import json
import math
import random
import statistics as st
import sys

ref = json.load(open(sys.argv[1]))


def nll(r):
    return [-p["actual"] if p["actual"] is not None else None for c in r["chunks"] for p in c["pos"]]


nr = nll(ref)
print(
    "reference",
    sys.argv[1],
    "NLL %.5f ppl %.4f" % (st.mean(x for x in nr if x is not None), math.exp(st.mean(x for x in nr if x is not None))),
)
for f in sys.argv[2:]:
    r = json.load(open(f))
    nx = nll(r)
    d = [a - b for a, b in zip(nx, nr) if a is not None and b is not None]
    random.seed(0)
    bs = sorted(st.mean(random.choices(d, k=len(d))) for _ in range(2000))
    kl, top, n = [], 0, 0
    for cr, cx in zip(ref["chunks"], r["chunks"]):
        for pr, px in zip(cr["pos"], cx["pos"]):
            fl = min(px["top"].values())
            kl.append(sum(math.exp(lp) * (lp - px["top"].get(k, fl)) for k, lp in pr["top"].items()))
            top += max(pr["top"], key=pr["top"].get) == max(px["top"], key=px["top"].get)
            n += 1
    m = st.mean(x for x in nx if x is not None)
    print(
        "%-28s NLL %.5f ppl %.4f  dNLL vs ref %+.5f [%+.5f, %+.5f]  KL %.3e  top-1 %.4f"
        % (f, m, math.exp(m), st.mean(d), bs[50], bs[1949], st.mean(kl), top / n)
    )
