"""Bigger evaluation sets from wiki.json: (1) every non-overlapping 1,500-token chunk of the four languages,
(2) one 16,000-token chunk per language for the depth analysis.  python make_texts_big.py MODEL_DIR big.json long.json wiki.json"""

import json
import sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
W = json.load(open(sys.argv[4]))
big, long = {"chunks": [], "prompts": []}, {"chunks": [], "prompts": []}
for lang, text in W.items():
    ids = tok(text)["input_ids"]
    long["chunks"].append({"lang": lang, "ids": ids[:16000]})
    for i in range(0, len(ids) - 1500, 1500):
        big["chunks"].append({"lang": lang, "ids": ids[i : i + 1500]})
json.dump(big, open(sys.argv[2], "w"))
json.dump(long, open(sys.argv[3], "w"))
print(
    len(big["chunks"]),
    "short chunks,",
    sum(len(c["ids"]) for c in big["chunks"]),
    "tokens;",
    [len(c["ids"]) for c in long["chunks"]],
    "long",
)
