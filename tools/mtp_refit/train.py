# Re-distills the dense part of the MTP layer on this engine's output distribution (compact dumps). Recipe
# as ursuciprian/Jrouth: experts, embedding, head and indexer frozen; soft CE to the target top-20
# (renormalized, restricted to the draft shortlist) at every depth of a 3-step teacher-forced unroll; AdamW,
# warmup, cosine. Held-out eval: acceptance over simulated MTP-3 rounds + per depth.
import argparse
import json
import math
import os
import random
import sys
import time
import collections
import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_model import load, export
from data import sequences, load_seq

SDPA = os.environ.get("MTP_REFIT_SDPA", "1") == "1"
CHUNK = 1024
SHORT = os.environ.get("MTP_REFIT_SHORTLIST", "")  # <artifact>.draft_shortlist.txt


def load_short(dev, V):
    ids = torch.tensor(sorted({int(l.split("#")[0]) for l in open(SHORT) if l.split("#")[0].strip()}), device=dev)
    lookup = torch.full((V,), -1, dtype=torch.long, device=dev)
    lookup[ids] = torch.arange(ids.numel(), device=dev)
    return ids, lookup


def soft_ce(sample, head_s, tk_ids, tk_lp, lookup):
    logq = (sample.to(head_s.dtype) @ head_s.T).float().log_softmax(-1)
    j = lookup[tk_ids.long()]
    p = tk_lp.float().softmax(-1) * (j >= 0)
    return -(p * logq.gather(1, j.clamp(min=0))).sum()


def window_loss(model, embed, head_s, lookup, toks, h, tki, tkl, P, s, e, depth, weights, count_from):
    """Anchors s..e-1 (rows of h); loss on anchors >= count_from with x[t+1] generated."""
    N = toks.shape[0]
    ids = torch.cat([toks, toks.new_zeros(depth + 1)])[s : e + depth + 1]
    samples = model.unroll(embed, ids, h[s:e], s, depth, ckpt=True, sdpa=SDPA)
    t = torch.arange(s, e, device=h.device)
    total, counts, ces = 0.0, [], []
    for k in range(depth):
        row = t + 1 + k
        valid = (t >= count_from) & (t + 1 >= P) & (row <= N - 2)
        idx = valid.nonzero().flatten()
        counts.append(int(idx.numel()))
        if not idx.numel():
            ces.append(0.0)
            continue
        r = row[idx]
        loss = 0.0
        for c0 in range(0, idx.numel(), CHUNK):  # chunks: [n, 117k] fp32 logits per depth do not fit at n ~ 3500
            sl = slice(c0, c0 + CHUNK)
            loss = loss + checkpoint(
                soft_ce, samples[k][(idx - s)[sl]], head_s, tki[r[sl]], tkl[r[sl]], lookup, use_reentrant=False
            )
        ces.append(float(loss.detach()))
        total = total + weights[k] * loss
    return total, counts, ces


@torch.no_grad()
def evaluate(model, embed, head_s, short, lookup, sets, depth, dev, max_seq=0):
    model.eval()
    agg = collections.defaultdict(
        lambda: {
            "rounds": 0,
            "pp": [0] * depth,
            "tf_ok": [0] * depth,
            "tf_n": [0] * depth,
            "t1": [0.0] * depth,
            "seqs": 0,
        }
    )
    for dump, gen, meta in sets:
        for n_seq, (i, d, P, m) in enumerate(sequences(dump, gen, meta)):
            if max_seq and n_seq >= max_seq:
                break
            toks, h, tki, tkl = load_seq(d, dev)
            N = toks.shape[0]
            n = h.shape[0]  # anchors 0..N-2
            ids = torch.cat([toks, toks.new_zeros(depth + 1)])
            with torch.autocast(h.device.type, dtype=torch.bfloat16):
                sm = model.unroll(embed, ids, h, 0, depth, sdpa=SDPA)
            dr, qx_all = [], []
            for k, x in enumerate(sm):  # per chunk: draft argmax and q at the target's top-k ids (rows t+1+k)
                am, qg = [], []
                rows = (torch.arange(n, device=dev) + 1 + k).clamp(max=n - 1)
                jj = lookup[tki[rows].long()]
                for c0 in range(0, n, CHUNK):
                    lg = (x[c0 : c0 + CHUNK].to(head_s.dtype) @ head_s.T).float()
                    am.append(lg.argmax(-1))
                    qg.append(lg.softmax(-1).gather(1, jj[c0 : c0 + CHUNK].clamp(min=0)) * (jj[c0 : c0 + CHUNK] >= 0))
                dr.append(short[torch.cat(am)])
                qx_all.append(torch.cat(qg))
            key = (m.get("cat", "?"), "on" if m.get("think") else "off")
            a = agg[key]
            a["seqs"] += 1
            for k in range(depth):
                t = torch.arange(max(P - 1, 0), n - 1 - k, device=dev)  # target x[t+2+k] exists, row t+1+k <= N-2
                if not t.numel():
                    continue
                a["tf_ok"][k] += int((dr[k][t] == toks[t + 2 + k]).sum())
                a["tf_n"][k] += t.numel()
                p = tkl[t + 1 + k].float().exp()
                a["t1"][k] += float(torch.minimum(p, qx_all[k][t]).sum())
            t, D = P - 1, [x.tolist() for x in dr]
            T = toks.tolist()
            while t + 2 < N and t < n:
                acc = 0
                while acc < depth and t + 2 + acc < N and D[acc][t] == T[t + 2 + acc]:
                    a["pp"][acc] += 1
                    acc += 1
                a["rounds"] += 1
                t += acc + 1
    out = {}
    for key, a in sorted(agg.items()):
        r = max(a["rounds"], 1)
        out["/".join(key)] = {
            "seqs": a["seqs"],
            "rounds": a["rounds"],
            "accept": sum(a["pp"]) / (depth * r),
            "tok_per_round": 1 + sum(a["pp"]) / r,
            "pp": [v / r for v in a["pp"]],
            "tf": [o / max(n, 1) for o, n in zip(a["tf_ok"], a["tf_n"])],
            "t1": [v / max(n, 1) for v, n in zip(a["t1"], a["tf_n"])],
        }
    model.train()
    return out


def parse_sets(spec):
    out = []
    for x in spec or []:
        p = x.split(":")
        out.append((p[0], p[1], p[2] if len(p) > 2 else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="dump_dir:gen.jsonl:meta.jsonl")
    ap.add_argument("--heldout", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--weights", default="1,1,1")
    ap.add_argument(
        "--window",
        type=int,
        default=4096,
        help="whole sequences only: a window without position 0 loses the attention sink (T194)",
    )
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--anchors-per-step", type=int, default=8192)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--experts-impl", default=None)
    ap.add_argument("--eval-max-seq", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--refit", default=None)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=194)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(os.environ.get("MTP_REFIT_DEVICE", "cuda"))
    if dev.type == "cpu":
        torch.set_num_threads(int(os.environ.get("MTP_REFIT_THREADS", "2")))
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    model, embed, head = load(dev, train=not a.eval_only, experts_impl=a.experts_impl, refit=a.refit)
    short, lookup = load_short(dev, head.shape[0])
    head_s = head[short].contiguous()
    del head
    weights = [float(x) for x in a.weights.split(",")]
    held = parse_sets(a.heldout)
    res = {"args": vars(a)}
    if held:
        t0 = time.time()
        res["before"] = evaluate(model, embed, head_s, short, lookup, held, a.depth, dev, a.eval_max_seq)
        print("before", json.dumps(res["before"]), f"{time.time() - t0:.0f}s", flush=True)
    if a.eval_only:
        json.dump(res, open(f"{a.out}/eval.json", "w"), indent=1)
        return
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable {len(params)} tensors {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    seqs = [s for spec in parse_sets(a.data) for s in sequences(*spec)]
    # one window per whole sequence: windows that do not start at position 0 lose the <|im_start|> attention
    # sink and the MTP attention collapses (measured: soft CE 20-27 nats vs 0.8-1.7 on the whole sequence)
    jobs = []
    for i, d, P, m in seqs:
        n = os.path.getsize(f"{d}/topk.bin") // (4 + 160)
        if n > a.window:
            print(f"skip {d}: {n} anchors > window {a.window}", flush=True)
            continue
        jobs.append((d, P, 0, n, 0))
    est = sum(max(0, e - (P - 1)) for d, P, s, e, cf in jobs)
    total_steps = a.max_steps or max(1, int(a.epochs * est / a.anchors_per_step))
    print(f"seqs {len(seqs)} windows {len(jobs)} anchors~{est} steps {total_steps}", flush=True)
    log = open(f"{a.out}/train.jsonl", "a")
    step, seen, acc_ce, acc_n, t0 = 0, 0, [0.0] * a.depth, [0] * a.depth, time.time()
    done = False
    ep = 0
    while not done:
        order = jobs[:]
        rng.shuffle(order)
        for d, P, s, e, cf in order:
            toks, h, tki, tkl = load_seq(d, dev)
            with torch.autocast(h.device.type, dtype=torch.bfloat16):
                loss, counts, ces = window_loss(
                    model, embed, head_s, lookup, toks, h, tki, tkl, P, s, e, a.depth, weights, cf
                )
            if not counts[0]:
                continue
            (loss / a.anchors_per_step).backward()
            seen += counts[0]
            for k in range(a.depth):
                acc_ce[k] += ces[k]
                acc_n[k] += counts[k]
            if seen < a.anchors_per_step:
                continue
            step += 1
            lr = (
                a.lr
                * min(1.0, step / a.warmup)
                * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step / total_steps, 1))))
            )
            for g in opt.param_groups:
                g["lr"] = lr
            gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip))
            opt.step()
            opt.zero_grad(set_to_none=True)
            rec = {
                "step": step,
                "lr": lr,
                "gn": round(gn, 4),
                "ce": [round(c / max(n, 1), 4) for c, n in zip(acc_ce, acc_n)],
                "n": acc_n,
                "s": round(time.time() - t0, 1),
                "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            seen, acc_ce, acc_n = 0, [0.0] * a.depth, [0] * a.depth
            if step >= total_steps:
                done = True
                break
        ep += 1
    opt.zero_grad(set_to_none=True)
    export(model, f"{a.out}/mtp_refit.safetensors", {"steps": step, **{k: v for k, v in vars(a).items()}})
    if held:
        res["after"] = evaluate(model, embed, head_s, short, lookup, held, a.depth, dev, a.eval_max_seq)
        print("after", json.dumps(res["after"]), flush=True)
    res["steps"] = step
    json.dump(res, open(f"{a.out}/eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
