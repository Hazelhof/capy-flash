"""Forks in teacher forcing: positions where the single-GPU reference is nearly tied between its top-2 tokens.
Agreement of each TP2 arm with the reference's top-1 there, paired McNemar between the arms."""
import json, math, sys
ref = json.load(open(sys.argv[1])); arms = sys.argv[2:]
R = [p for c in ref["chunks"] for p in c["pos"]]
A = {a: [p for c in json.load(open(a))["chunks"] for p in c["pos"]] for a in arms}
def top(p): return max(p["top"], key=p["top"].get)
def margin(p):
    v = sorted(p["top"].values(), reverse=True); return v[0] - v[1] if len(v) > 1 else 9
for thr in (0.05, 0.1, 0.25, 0.5):
    idx = [i for i, p in enumerate(R) if margin(p) < thr]
    agree = {a: [top(A[a][i]) == top(R[i]) for i in idx] for a in arms}
    line = f"margin<{thr:<4}: {len(idx):5d} forks"
    for a in arms: line += f" | {a.split('/')[-1][:-5]} {sum(agree[a]) / len(idx):.3f}"
    if len(arms) == 2:
        x, y = agree[arms[0]], agree[arms[1]]
        b = sum(1 for u, v in zip(x, y) if u and not v); c = sum(1 for u, v in zip(x, y) if v and not u)
        z = (abs(b - c) - 1) / math.sqrt(b + c) if b + c else 0
        line += f" | only-first {b} only-second {c} McNemar p {math.erfc(z / math.sqrt(2)):.3f}"
    print(line)
