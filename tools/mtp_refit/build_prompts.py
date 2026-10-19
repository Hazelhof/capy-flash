# Prompts for the on-policy MTP data and disjoint per-language held-out sets, fixed seed.
# it: alpaca-gpt4-it + evol-instruct-it (non-code); code-it: evol-it coding requests; en: UltraChat test_gen;
# code-en: Magicoder OSS-Instruct. 30 % with reasoning on.
import json
import random
import re
import collections
import pandas as pd
from transformers import AutoTokenizer
import os

C = os.environ["CORPUS"]  # alpaca-gpt4-it.json, evol-it.json, ultrachat-test_gen.parquet, magicoder.jsonl
O = os.environ.get("OUT", ".")
tok = AutoTokenizer.from_pretrained(os.environ["MTP_REFIT_CHECKPOINT"])
rng = random.Random(193)
CODE = re.compile(
    r"(\bcodice\b|\bpython\b|\bjava\b|javascript|c\+\+|\bc#|\bsql\b|\bhtml\b|\bcss\b|\bbash\b|"
    r"\bscript\b|\bregex\b|\brust\b|\bgolang\b|\bphp\b|\bkotlin\b|\bmatlab\b|\blatex\b|```)",
    re.I,
)


def ok(p, lo=20, hi=2500):
    return isinstance(p, str) and lo <= len(p.strip()) <= hi


def first_human(rec):
    for t in rec["conversations"]:
        if t["from"] == "human":
            return t["value"].strip()


alp = [first_human(r) for r in json.load(open(f"{C}/alpaca-gpt4-it.json"))]
evo = [first_human(r) for r in json.load(open(f"{C}/evol-it.json"))]
it_plain = [p for p in alp + evo if ok(p) and not CODE.search(p)]
it_code = [p for p in evo + alp if ok(p) and CODE.search(p)]
en = [p for p in pd.read_parquet(f"{C}/ultrachat-test_gen.parquet").prompt if ok(p)]
code_en = [json.loads(l)["problem"].strip() for l in open(f"{C}/magicoder.jsonl")]
code_en = [p for p in code_en if ok(p)]
pools = {"it": it_plain, "code-it": it_code, "en": en, "code-en": code_en}
N_TRAIN = {"it": 800, "code-it": 300, "en": 700, "code-en": 550}
N_HELD = 40
train, held = [], {}
for k, pool in pools.items():
    pool = list(dict.fromkeys(pool))
    rng.shuffle(pool)
    held[k] = pool[:N_HELD]
    train += [{"cat": k, "prompt": p} for p in pool[N_HELD : N_HELD + N_TRAIN[k]]]
    print(k, "pool", len(pool))
rng.shuffle(train)
for r in train:
    r["think"] = rng.random() < 0.30
out = {False: [], True: []}
for r in train:
    ids = tok.apply_chat_template(
        [{"role": "user", "content": r["prompt"]}],
        add_generation_prompt=True,
        enable_thinking=r["think"],
        tokenize=True,
    )
    ids = list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)
    if len(ids) > 1500:
        continue
    r["prompt_tokens"] = len(ids)
    out[r["think"]].append((ids, r))
for think, rows in out.items():
    tag = "on" if think else "off"
    with open(f"{O}/train_{tag}.ids", "w") as f, open(f"{O}/train_{tag}.meta.jsonl", "w") as g:
        for ids, r in rows:
            f.write(" ".join(map(str, ids)) + "\n")
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(tag, len(rows), collections.Counter(r["cat"] for _, r in rows))
json.dump(held, open(f"{O}/heldout.json", "w"), ensure_ascii=False, indent=0)
