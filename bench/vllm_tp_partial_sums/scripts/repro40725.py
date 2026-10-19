"""Issue #40725 reproduction on 2x RTX 5070 Ti: Qwen3.6-35B-A3B NVFP4, the issue's Spanish 40-stanza poem (and Italian,
Portuguese variants), sampled exactly as in the issue (T 0.7, top_p 0.9, seed 99) plus greedy.
  python repro40725.py MODEL out.json        env: EP=1 -> --enable-expert-parallel, DTYPE=float16 -> whole model in FP16, VLLM_TP_FP16_PARTIAL_SUMS"""

import json
import os
import sys
import time
from vllm import LLM, SamplingParams

P = {
    "es": "Escribe un poema de exactamente 40 estrofas (cuartetos de 4 versos cada una) sobre el viaje de un astronauta solitario que orbita Júpiter contemplando las tormentas del planeta. Usa lenguaje rico en español, con metáforas y sin repetirte. Numera cada estrofa con romanos (I, II, III...). No incluyas ningún razonamiento, solo el poema.",
    "it": "Scrivi una poesia di esattamente 40 strofe (quartine di 4 versi ciascuna) sul viaggio di un astronauta solitario che orbita attorno a Giove contemplando le tempeste del pianeta. Usa un linguaggio ricco in italiano, con metafore e senza ripeterti. Numera ogni strofa con numeri romani (I, II, III...). Non includere alcun ragionamento, solo la poesia.",
    "pt": "Escreva um poema de exatamente 40 estrofes (quadras de 4 versos cada) sobre a viagem de um astronauta solitário que orbita Júpiter contemplando as tempestades do planeta. Use uma linguagem rica em português, com metáforas e sem se repetir. Numere cada estrofe com algarismos romanos (I, II, III...). Não inclua nenhum raciocínio, apenas o poema.",
    "en": "Write a poem of exactly 40 stanzas (quatrains of 4 lines each) about the journey of a lonely astronaut orbiting Jupiter, contemplating the planet's storms. Use rich language, with metaphors and without repeating yourself. Number each stanza with Roman numerals (I, II, III...). Do not include any reasoning, only the poem.",
}


def main():
    model, out = sys.argv[1], sys.argv[2]
    llm = LLM(
        model=model,
        tensor_parallel_size=2,
        max_model_len=12288,
        dtype=os.environ.get("DTYPE", "auto"),
        gpu_memory_utilization=0.90,
        seed=0,
        enable_expert_parallel=os.environ.get("EP") == "1",
        max_num_seqs=8,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    tok = llm.get_tokenizer()
    msgs = {
        k: tok.apply_chat_template(
            [{"role": "user", "content": v}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for k, v in P.items()
    }
    res = {
        "ep": os.environ.get("EP", "0"),
        "dtype": os.environ.get("DTYPE", "auto"),
        "fp16": os.environ.get("VLLM_TP_FP16_PARTIAL_SUMS", "0"),
        "runs": [],
    }
    for mode, sp in (
        ("sampled", SamplingParams(temperature=0.7, top_p=0.9, seed=99, max_tokens=8000)),
        ("greedy", SamplingParams(temperature=0, max_tokens=8000)),
    ):
        langs = list(msgs)
        t0 = time.time()
        o = llm.generate([msgs[k] for k in langs], sp)
        for k, r in zip(langs, o):
            res["runs"].append(
                {
                    "mode": mode,
                    "lang": k,
                    "text": r.outputs[0].text,
                    "ntok": len(r.outputs[0].token_ids),
                    "finish": r.outputs[0].finish_reason,
                }
            )
        print(mode, "done", round(time.time() - t0), "s", flush=True)
    json.dump(res, open(out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
