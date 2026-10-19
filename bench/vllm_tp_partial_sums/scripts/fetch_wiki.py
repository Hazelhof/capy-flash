"""Download the evaluation text: two long history articles per language from Wikipedia (CC BY-SA 4.0), plain text
via the MediaWiki API, into wiki.json.  python fetch_wiki.py wiki.json"""

import json
import sys
import urllib.parse
import urllib.request

A = {
    "en": ["Roman_Empire", "World_War_I"],
    "it": ["Storia_d'Italia", "Repubblica_di_Venezia"],
    "es": ["Historia_de_España", "Imperio_español"],
    "fr": ["Histoire_de_France", "Révolution_française"],
}
out = {}
for lang, titles in A.items():
    txt = ""
    for t in titles:
        q = urllib.parse.urlencode(
            {
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "titles": t.replace("_", " "),
                "format": "json",
                "redirects": 1,
            }
        )
        req = urllib.request.Request(
            f"https://{lang}.wikipedia.org/w/api.php?{q}", headers={"User-Agent": "vllm-tp-partial-sums-eval/1.0"}
        )
        page = next(iter(json.load(urllib.request.urlopen(req, timeout=60))["query"]["pages"].values()))
        txt += page.get("extract", "") + "\n\n"
    out[lang] = txt
    print(lang, len(txt), "chars")
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False)
