# Before/after table of the held-out eval (train.py eval.json): acceptance over simulated MTP-3 rounds, tokens
# per round, per position; and T=1 acceptance (sum min(p,q) over the top-20) per depth.
import json
import sys

r = json.load(open(sys.argv[1]))
b, a = r["before"], r.get("after", {})
print(
    f"{'category':14s} {'seq':>4s}  accept before→after   tok/round before→after   per pos. after        T=1 before→after (depth 0/1/2)"
)
for k in sorted(b):
    x, y = b[k], a.get(k)
    if not y:
        continue
    print(
        f"{k:14s} {x['seqs']:4d}  {x['accept']:.3f} → {y['accept']:.3f} ({y['accept'] - x['accept']:+.3f})   "
        f"{x['tok_per_round']:.2f} → {y['tok_per_round']:.2f}   {' '.join(f'{v:.3f}' for v in y['pp'])}   "
        f"{' '.join(f'{v:.2f}' for v in x['t1'])} → {' '.join(f'{v:.2f}' for v in y['t1'])}"
    )
