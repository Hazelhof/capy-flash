"""Counts #40725-style corruption in the poems: pattern hits (doubled words, self-glued words, split prefixes,
non-Latin script, digit runs) and words unknown to a dictionary of the language (reviewed by hand)."""
import json, re, sys, unicodedata
from spellchecker import SpellChecker
SC = {l: SpellChecker(language=l) for l in ("es", "it", "pt", "en")}
WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
def analyse(text, lang):
    hits = []
    words = WORD.findall(text)
    for a, b in zip(words, words[1:]):
        if a.lower() == b.lower() and len(a) > 2: hits.append(f"doubled:{a} {b}")
        elif len(a) >= 2 and b.lower().startswith(a.lower()) and len(b) > len(a) + 1 and a.lower() not in SC[lang]: hits.append(f"split:{a} {b}")
    for w in words:
        m = re.search(r"([a-zà-ÿ]{3,})\1", w.lower())
        if m and w.lower() not in SC[lang]: hits.append(f"selfglued:{w}")
    for ch in set(text):
        if ch.isalpha() and not unicodedata.name(ch, "").startswith(("LATIN", "GREEK")): hits.append(f"script:{ch}")
    if re.search(r"\d{6,}", text): hits.append("digitrun")
    lw = [w for w in words if not w[0].isupper() and len(w) > 2]
    unk = sorted(SC[lang].unknown([w.lower() for w in lw]))
    return hits, unk
out = {}
for arm in sys.argv[1:]:
    r = json.load(open(f"repro-{arm}.json"))
    for x in r["runs"]:
        hits, unk = analyse(x["text"], x["lang"])
        out[f"{arm}|{x['mode']}|{x['lang']}"] = {"hits": hits, "unknown": unk}
        print(f"{arm:8s} {x['mode']:7s} {x['lang']}  pattern hits {len(hits):2d}  unknown words {len(unk):3d}  {'; '.join(hits)[:150]}")
json.dump(out, open("corrupt.json", "w"), ensure_ascii=False, indent=0)
