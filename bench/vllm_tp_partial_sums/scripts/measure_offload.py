"""One vLLM configuration: teacher-forced prompt logprobs on fixed chunks, greedy generations, decode speed.
  python measure.py MODEL TP texts.json out.json     (VLLM_TP_FP16_PARTIAL_SUMS from the environment)"""
import json, os, sys, time
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
def main():
    model, tp, tf, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    T = json.load(open(tf))
    llm = LLM(model=model, tensor_parallel_size=tp, dtype=os.environ.get("DTYPE", "bfloat16"), max_model_len=4096, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.80")), seed=0,
              enable_expert_parallel=os.environ.get("EP") == "1", cpu_offload_gb=float(os.environ.get("OFFLOAD_GB", "0")), **({"linear_backend": os.environ["LINEAR_BACKEND"]} if os.environ.get("LINEAR_BACKEND") else {}), max_num_batched_tokens=2048, max_num_seqs=16, **({"limit_mm_per_prompt": {"image": 0, "video": 0}} if os.environ.get("MM0") else {}))
    res = {"ep": os.environ.get("EP", "0"), "dtype": os.environ.get("DTYPE", "bfloat16"), "tp": tp, "fp16_partials": os.environ.get("VLLM_TP_FP16_PARTIAL_SUMS", "0"), "chunks": [], "gen": [], "speed": {}}
    sp = SamplingParams(max_tokens=1, prompt_logprobs=20, temperature=0)
    outs = llm.generate([TokensPrompt(prompt_token_ids=c["ids"]) for c in T["chunks"]], sp)
    for c, o in zip(T["chunks"], outs):
        pos = []
        for t, lp in zip(c["ids"][1:], o.prompt_logprobs[1:]):
            pos.append({"actual": lp[t].logprob if t in lp else None, "top": {str(k): v.logprob for k, v in lp.items()}})
        res["chunks"].append({"lang": c["lang"], "pos": pos})
    if os.environ.get("LOGPROBS_ONLY"):
        json.dump(res, open(out, "w")); print("done", out, "logprobs only"); return
    g = llm.generate([p["text"] for p in T["prompts"]], SamplingParams(max_tokens=384, temperature=0))
    res["gen"] = [{"lang": p["lang"], "ids": list(o.outputs[0].token_ids), "text": o.outputs[0].text} for p, o in zip(T["prompts"], g)]
    for bs in (1, 16):
        ps = [T["prompts"][i % len(T["prompts"])]["text"] for i in range(bs)]
        llm.generate(ps, SamplingParams(max_tokens=32, temperature=0))  # warm
        t0 = time.time(); o = llm.generate(ps, SamplingParams(max_tokens=256, temperature=0, ignore_eos=True)); dt = time.time() - t0
        res["speed"][f"bs{bs}"] = sum(len(x.outputs[0].token_ids) for x in o) / dt
    json.dump(res, open(out, "w")); print("done", out, res["speed"])


if __name__ == "__main__":
    main()
