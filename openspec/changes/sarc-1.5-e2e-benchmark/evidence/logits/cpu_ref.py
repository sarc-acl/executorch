# CPU float reference for the next-token logits after prompt_check.txt.
# Model: ExecuTorch examples/models/llama Transformer (dev/1.5 clone), Meta checkpoint
# consolidated.00.pth (bf16 weights, loaded via mmap, assign=True).
# --compute fp32: bf16 weights up-cast to fp32 layer by layer (fp32 activations/accumulation).
# --compute bf16: everything in bf16 (what a plain bf16 reference would do).
# --tokens runner: C++ tiktoken (same library as llama_main), no BOS (llama_main --num_bos default 0,
#                  e2e.sh does not pass it) -> 1972 tokens, exactly what the GPU runs saw.
#          runner_bos: same + BOS 128000.   py_bos: Python tiktoken encoding + BOS.
import argparse, json, time, sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--compute", choices=["fp32", "bf16"], default="fp32")
ap.add_argument("--tokens", choices=["runner", "runner_bos", "py_bos"], default="runner")
ap.add_argument("--out", required=True)
args = ap.parse_args()

ROOT = "/mnt/linux-share/models/llama-3.1-8b/original/"
PROMPT = "/home/doremy/Desktop/sarc-acl/.artifacts/e2e-1.5-2026-09-28/tools/prompt_check.txt"
text = open(PROMPT, encoding="utf-8").read()

import pytorch_tokenizers as ptk
from pytorch_tokenizers.tiktoken import TiktokenTokenizer

cpp = ptk.CppTiktoken()
cpp.load(ROOT + "tokenizer.model")
if args.tokens == "runner":
    ids = list(cpp.encode(text, 0, 0))
elif args.tokens == "runner_bos":
    ids = list(cpp.encode(text, 1, 0))
else:
    ids = TiktokenTokenizer(ROOT + "tokenizer.model").encode(text, bos=True, eos=False)
print(f"tokens={args.tokens} n={len(ids)} first={ids[:3]} last={ids[-4:]}", flush=True)

from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.rope import Rope

params = json.load(open(ROOT + "params.json"))
margs = ModelArgs(**params, max_seq_len=2048, max_context_len=2048, use_kv_cache=False)
assert margs.rope_freq_base == 500000.0 and margs.use_scaled_rope and margs.rope_scale_factor == 8
with torch.device("meta"):
    model = construct_transformer(margs)
sd = torch.load(ROOT + "consolidated.00.pth", mmap=True, weights_only=True, map_location="cpu")
sd = {k: v for k, v in sd.items() if not k.endswith("rope.freqs")}
missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
print("missing:", missing, "unexpected:", unexpected, flush=True)
assert not missing and not unexpected
# Non-persistent buffers: rebuild on CPU if they were created on meta.
rope = Rope(margs)
model.rope = rope
mask = torch.tril(torch.ones(2048, 2048, dtype=torch.bool))
for layer in model.layers:
    layer.attention.rope = rope
    if layer.attention.mask.is_meta:
        layer.attention.mask = mask
for n, b in model.named_buffers():
    assert not b.is_meta, n
for n, p in model.named_parameters():
    assert not p.is_meta, n
model.eval()

if args.compute == "fp32":
    model.norm.float()
    model.output.float()

    def pre(mod, inp):
        mod._bf16 = {n: p.data for n, p in mod.named_parameters()}
        for n, p in mod.named_parameters():
            p.data = p.data.float()

    def post(mod, inp, out):
        for n, p in mod.named_parameters():
            p.data = mod._bf16[n]
        del mod._bf16

    for layer in model.layers:
        layer.register_forward_pre_hook(pre)
        layer.register_forward_hook(post)

tok = torch.tensor([ids], dtype=torch.long)
torch.set_num_threads(torch.get_num_threads())
t0 = time.time()
with torch.no_grad():
    h = model.tok_embeddings(tok)
    if args.compute == "fp32":
        h = h.float()
    logits = model(h=h)
    if isinstance(logits, tuple):
        logits = logits[0]
logits = logits.float().reshape(-1)
print(f"forward {time.time() - t0:.1f}s shape={tuple(logits.shape)}", flush=True)

OTH, BUL = 6062, 45647
assert cpp.decode(OTH) == " otherwise" and cpp.decode(BUL) == " bullying"
probs = torch.softmax(logits.double(), -1)
top = torch.topk(logits, 10)
rows = []
for v, i in zip(top.values.tolist(), top.indices.tolist()):
    rows.append({"id": i, "tok": cpp.decode(i), "logit": v, "prob": probs[i].item()})
res = {
    "compute": args.compute, "weights": "bf16 (Meta consolidated.00.pth)",
    "tokens": args.tokens, "n_tokens": len(ids), "last_ids": ids[-8:],
    "top10": rows,
    "otherwise": {"id": OTH, "logit": logits[OTH].item(), "prob": probs[OTH].item(),
                  "rank": int((logits > logits[OTH]).sum()) + 1},
    "bullying": {"id": BUL, "logit": logits[BUL].item(), "prob": probs[BUL].item(),
                 "rank": int((logits > logits[BUL]).sum()) + 1},
}
res["dlogit_otherwise_minus_bullying"] = res["otherwise"]["logit"] - res["bullying"]["logit"]
res["dprob_otherwise_minus_bullying"] = res["otherwise"]["prob"] - res["bullying"]["prob"]
json.dump(res, open(args.out + ".json", "w"), indent=1, ensure_ascii=False)
torch.save({"ids": ids, "logits": logits}, args.out + ".pt")
for r in rows[:5]:
    print(f"{r['id']:>7} {r['tok']!r:>16} logit={r['logit']:.4f} p={r['prob']:.4f}")
print("otherwise", res["otherwise"], "\nbullying", res["bullying"])
print("dlogit", res["dlogit_otherwise_minus_bullying"], "dprob", res["dprob_otherwise_minus_bullying"])
