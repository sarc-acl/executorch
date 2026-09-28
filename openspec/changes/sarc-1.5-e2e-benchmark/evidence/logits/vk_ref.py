# Vulkan 8da4w next-token logits after prompt_check.txt, via the ExecuTorch Python runtime
# of the dev/1.5 clone venv. One prefill call: forward(tokens[1,1972], input_pos=[0]) ->
# last-position logits [1,128256]. Tokens exactly as llama_main encodes them (C++ tiktoken,
# --num_bos default 0 -> no BOS). Mode is selected by env (ET_VK_FORCE_TILED_LINEAR).
import json, os, sys, time
import torch

out = sys.argv[1]
PTE = "/mnt/linux-share/models/llama-3.1-8b/exported/llama3_1-8b_vulkan_8da4w.pte"
ROOT = "/mnt/linux-share/models/llama-3.1-8b/original/"
PROMPT = "/home/doremy/Desktop/sarc-acl/.artifacts/e2e-1.5-2026-09-28/tools/prompt_check.txt"

import pytorch_tokenizers as ptk
cpp = ptk.CppTiktoken()
cpp.load(ROOT + "tokenizer.model")
ids = list(cpp.encode(open(PROMPT, encoding="utf-8").read(), 0, 0))
assert len(ids) == 1972, len(ids)

from executorch.runtime import Runtime
rt = Runtime.get()
prog = rt.load_program(PTE)
fwd = prog.load_method("forward")
tok = torch.tensor([ids], dtype=torch.long)
pos = torch.tensor([0], dtype=torch.long)
t0 = time.time()
logits = fwd.execute([tok, pos])[0].float().reshape(-1).clone()
dt = time.time() - t0

OTH, BUL = 6062, 45647
probs = torch.softmax(logits.double(), -1)
top = torch.topk(logits, 10)
rows = [{"id": i, "tok": cpp.decode(i), "logit": v, "prob": probs[i].item()}
        for v, i in zip(top.values.tolist(), top.indices.tolist())]
res = {
    "pte": PTE, "env": {k: v for k, v in os.environ.items() if k.startswith(("ET_VK", "ETVK"))},
    "n_tokens": len(ids), "prefill_s": dt, "top10": rows,
    "otherwise": {"id": OTH, "logit": logits[OTH].item(), "prob": probs[OTH].item(),
                  "rank": int((logits > logits[OTH]).sum()) + 1},
    "bullying": {"id": BUL, "logit": logits[BUL].item(), "prob": probs[BUL].item(),
                 "rank": int((logits > logits[BUL]).sum()) + 1},
}
res["dlogit_otherwise_minus_bullying"] = res["otherwise"]["logit"] - res["bullying"]["logit"]
res["dprob_otherwise_minus_bullying"] = res["otherwise"]["prob"] - res["bullying"]["prob"]
json.dump(res, open(out + ".json", "w"), indent=1, ensure_ascii=False)
torch.save({"ids": ids, "logits": logits}, out + ".pt")
print(json.dumps(res["env"]), f"prefill {dt:.2f}s")
for r in rows[:5]:
    print(f"{r['id']:>7} {r['tok']!r:>16} logit={r['logit']:.4f} p={r['prob']:.4f}")
print("otherwise", res["otherwise"], "\nbullying", res["bullying"])
print("dlogit", res["dlogit_otherwise_minus_bullying"], "dprob", res["dprob_otherwise_minus_bullying"])
