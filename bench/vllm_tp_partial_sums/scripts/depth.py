"""Does the TP2 error grow with context depth? Long 16k chunks: per position bucket, mean KL from the reference
and top-1 agreement at forks (reference top-2 margin < 0.25) for each arm."""
import json, math, statistics as st, sys
ref = json.load(open(sys.argv[1])); arms = {a: json.load(open(a)) for a in sys.argv[2:]}
B = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]
def top(p): return max(p["top"], key=p["top"].get)
def kl(pr, px):
    fl = min(px["top"].values()); return sum(math.exp(lp) * (lp - px["top"].get(k, fl)) for k, lp in pr["top"].items())
print("bucket        " + "".join(f"| {a.split('/')[-1][:-5]:>22s} " for a in arms))
for lo, hi in B:
    line = f"{lo:>5}-{hi:<6}  "
    for a, r in arms.items():
        k, f, n = [], 0, 0
        for cr, cx in zip(ref["chunks"], r["chunks"]):
            for i in range(lo, min(hi, len(cr["pos"]))):
                pr, px = cr["pos"][i], cx["pos"][i]; k.append(kl(pr, px))
                v = sorted(pr["top"].values(), reverse=True)
                if len(v) > 1 and v[0] - v[1] < 0.25: n += 1; f += top(pr) == top(px)
        line += f"| KL {st.mean(k):.2e} fork {f / max(n, 1):.3f} ({n:4d}) "
    print(line)
