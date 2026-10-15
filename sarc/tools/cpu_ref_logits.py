#!/usr/bin/env python3
"""cpu_ref_logits.py: the fp32 CPU reference logits for verify.sh's near-tie stage (near_tie.py, MERGE-PLAN M1).

Generalised from sarc-1.5-e2e-benchmark/evidence/logits/cpu_ref.py (--compute fp32 --tokens runner): the
ExecuTorch examples/models/llama Transformer with the Meta checkpoint (bf16 weights, mmap), up-cast to fp32 layer
by layer, fp32 activations and accumulation; next-token logits after the whole prompt. No GPU. The 8B model needs
about 19 GB of RAM and a few minutes. A reference depends on the model and the prompt only, never on a build, so
it is computed once per (model, prompt) and kept.

  cpu_ref_logits.py --model-dir <dir with consolidated.00.pth, params.json, tokenizer.model>
                    (--prompt <file> | --ids <file>) --out <prefix> [--write-ids <file>]
  -> <prefix>.json {"compute": "fp32", "n_tokens", "ids_sha256", "top10": [[id, logit], ...]} and <prefix>.f32
Tokens: as llama_main feeds them (C++ tiktoken, no BOS). --write-ids stores them (whitespace-separated) for
logits_probe. verify.sh expects <ref dir>/<model>-<prompt stem>.json, e.g. 8b-prompt_check.json.
Needs a Python environment with torch, executorch (examples.models.llama) and pytorch_tokenizers."""
import argparse, hashlib, json, os
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--prompt")
g.add_argument("--ids")
ap.add_argument("--out", required=True)
ap.add_argument("--write-ids")
a = ap.parse_args()
R = a.model_dir.rstrip("/") + "/"

if a.prompt:
    import pytorch_tokenizers as ptk
    cpp = ptk.CppTiktoken()
    cpp.load(R + "tokenizer.model")
    ids = list(cpp.encode(open(a.prompt, encoding="utf-8").read(), 0, 0))
else:
    ids = [int(x) for x in open(a.ids).read().split()]
if a.write_ids:
    open(a.write_ids, "w").write(" ".join(map(str, ids)) + "\n")
assert 0 < len(ids) <= 2048, len(ids)

from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import Rope

margs = ModelArgs(**json.load(open(R + "params.json")), max_seq_len=2048, max_context_len=2048, use_kv_cache=False)
with torch.device("meta"):
    model = construct_transformer(margs)
sd = torch.load(R + "consolidated.00.pth", mmap=True, weights_only=True, map_location="cpu")
sd = {k: v for k, v in sd.items() if not k.endswith("rope.freqs")}
missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
assert not missing and not unexpected, (missing, unexpected)
rope = Rope(margs)
model.rope = rope
mask = torch.tril(torch.ones(2048, 2048, dtype=torch.bool))
for layer in model.layers:
    layer.attention.rope = rope
    if layer.attention.mask.is_meta:
        layer.attention.mask = mask
assert not any(b.is_meta for b in model.buffers()) and not any(p.is_meta for p in model.parameters())
model.eval()
model.norm.float()
model.output.float()


def pre(mod, inp):  # up-cast one layer's bf16 weights to fp32 for its forward, then restore
    mod._bf16 = {n: p.data for n, p in mod.named_parameters()}
    for _, p in mod.named_parameters():
        p.data = p.data.float()


def post(mod, inp, out):
    for n, p in mod.named_parameters():
        p.data = mod._bf16[n]
    del mod._bf16


for layer in model.layers:
    layer.register_forward_pre_hook(pre)
    layer.register_forward_hook(post)
with torch.no_grad():
    out = model(h=model.tok_embeddings(torch.tensor([ids], dtype=torch.long)).float())
    out = out[0] if isinstance(out, tuple) else out
logits = out.float().reshape(-1, out.shape[-1])[-1].contiguous()
top = torch.topk(logits, 10)
res = {"compute": "fp32", "weights": "bf16 checkpoint up-cast to fp32", "model_dir": R, "tokens": "runner, no BOS",
       "n_tokens": len(ids), "ids_sha256": hashlib.sha256(" ".join(map(str, ids)).encode()).hexdigest(),
       "vocab": int(logits.numel()), "top10": [[int(i), float(v)] for v, i in zip(top.values.tolist(), top.indices.tolist())]}
json.dump(res, open(a.out + ".json", "w"), indent=1)
logits.numpy().astype("<f4").tofile(a.out + ".f32")
print(f"n_tokens={len(ids)} top1 {res['top10'][0]} top2 {res['top10'][1]} margin {res['top10'][0][1] - res['top10'][1][1]:.4f}")
