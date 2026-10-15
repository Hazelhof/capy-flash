"""Fixed multilingual evaluation texts from Wikipedia (CC BY-SA; two long history articles per language, fetched once into
wiki.json): 4 languages x 4 chunks of 1,500 tokens from the middle of the text, plus chat prompts for greedy generation.
  python make_texts.py MODEL_DIR texts.json wiki.json"""
import json, sys
from transformers import AutoTokenizer
PROMPTS = {
 "en": ["Explain in detail how a refrigerator works.", "Write a short story about a lighthouse keeper.", "List the pros and cons of remote work, then summarize."],
 "it": ["Spiega in dettaglio come funziona un frigorifero.", "Scrivi un breve racconto su un guardiano del faro.", "Elenca pro e contro del lavoro da remoto, poi riassumi."],
 "es": ["Explica en detalle cómo funciona un refrigerador.", "Escribe un cuento corto sobre un guardián de faro.", "Enumera los pros y contras del trabajo remoto y luego resume."],
 "fr": ["Explique en détail comment fonctionne un réfrigérateur.", "Écris une courte histoire sur un gardien de phare.", "Liste les avantages et inconvénients du télétravail, puis résume."],
}
tok = AutoTokenizer.from_pretrained(sys.argv[1]); out = {"chunks": [], "prompts": []}
W = json.load(open(sys.argv[3]))
for lang, text in W.items():
    ids = tok(text)["input_ids"]; mid = len(ids) // 2
    for i in range(4):
        out["chunks"].append({"lang": lang, "ids": ids[mid + i * 1500: mid + (i + 1) * 1500]})
    for p in PROMPTS[lang]:
        out["prompts"].append({"lang": lang, "text": tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)})
json.dump(out, open(sys.argv[2], "w")); print(len(out["chunks"]), "chunks", len(out["prompts"]), "prompts")
