"""Tails of the per-position deviation from the single-GPU reference: percentiles of KL and of |dlogprob| of the
actual token, and how many positions exceed thresholds, paired between two arms (McNemar on exceedances)."""

import json
import math
import sys

ref = json.load(open(sys.argv[1]))
A, B = (json.load(open(f)) for f in sys.argv[2:4])


def per_pos(r):
    kl, da = [], []
    for cr, cx in zip(ref["chunks"], r["chunks"]):
        for pr, px in zip(cr["pos"], cx["pos"]):
            fl = min(px["top"].values())
            kl.append(sum(math.exp(lp) * (lp - px["top"].get(k, fl)) for k, lp in pr["top"].items()))
            da.append(
                abs(pr["actual"] - px["actual"]) if pr["actual"] is not None and px["actual"] is not None else 0.0
            )
    return kl, da


ka, da = per_pos(A)
kb, db = per_pos(B)
n = len(ka)


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


print(f"positions {n}")
for name, va, vb in (("KL", ka, kb), ("|dlogprob actual|", da, db)):
    print(
        f"{name:18s} p50 {pct(va, 0.5):.2e}/{pct(vb, 0.5):.2e}  p99 {pct(va, 0.99):.3f}/{pct(vb, 0.99):.3f}  p99.9 {pct(va, 0.999):.3f}/{pct(vb, 0.999):.3f}  max {max(va):.2f}/{max(vb):.2f}   (first/second)"
    )
for thr in (0.05, 0.1, 0.25, 0.5, 1.0):
    ea = [x > thr for x in ka]
    eb = [x > thr for x in kb]
    b = sum(1 for u, v in zip(ea, eb) if u and not v)
    c = sum(1 for u, v in zip(ea, eb) if v and not u)
    z = (abs(b - c) - 1) / math.sqrt(b + c) if b + c else 0
    print(
        f"KL > {thr:<5}: first {sum(ea):5d}  second {sum(eb):5d}  (only first {b}, only second {c}, McNemar p {math.erfc(z / math.sqrt(2)):.3f})"
    )
