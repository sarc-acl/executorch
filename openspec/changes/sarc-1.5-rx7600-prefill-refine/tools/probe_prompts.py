#!/usr/bin/env python3
"""probe_prompts.py <out dir>: the real-text prompt set of the logits probe (owner decisions 2026-10-04).

32 prompts of different lengths (multiples of 128 tokens, 128 to 1920, so the coopmat and the fused SDPA kernels
serve them), cut as token windows from six real texts: the two real-text kit prompts and four documents of this
repository. The recipe is fixed here, not tuned: prompt i takes source i % 6, length 128 * (1 + (7 * i) % 15) and
start (131 * i) % (tokens - length); the token after the window is recorded as the true next token.
Writes prompts-<model>.txt (one prompt per line, token ids, for logits_probe) and prompts-<model>.json.
Run with a Python that has pytorch_tokenizers (/tool/pkg Python 3.12 on the RX 7600 host); read-only on everything else."""
import json, os, sys
import pytorch_tokenizers as ptk

ET = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
KIT = os.path.join(ET, "openspec/changes/sarc-1.5-e2e-benchmark/kit/prompts")
SOURCES = [os.path.join(KIT, "prompt_real_2048.txt"), os.path.join(KIT, "prompt_check.txt"),
           os.path.join(ET, "README.md"), os.path.join(ET, "CONTRIBUTING.md"),
           os.path.join(ET, "CODE_OF_CONDUCT.md"), os.path.join(ET, "docs/source/using-executorch-building-from-source.md")]
MODELS = {"1b": "llama-3.2-1b", "3b": "llama-3.2-3b", "8b": "llama-3.1-8b"}
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for m, d in MODELS.items():
    tok = ptk.CppTiktoken(); tok.load("/local/yanwen.xu/campaign-rx7600/.artifacts/models/tokenizer.model")  # one tokenizer for the three models (sha256 82e9d319...)
    ids = [list(tok.encode(open(s, encoding="utf-8").read(), 0, 0)) for s in SOURCES]
    rows = []
    for i in range(32):
        src = i % len(SOURCES); n = 128 * (1 + (7 * i) % 15)
        n = min(n, (len(ids[src]) - 1) // 128 * 128)
        start = (131 * i) % (len(ids[src]) - n)
        rows.append({"prompt": i, "source": os.path.relpath(SOURCES[src], ET), "source_tokens": len(ids[src]),
                     "start": start, "tokens": n, "next_token": ids[src][start + n],
                     "ids": ids[src][start:start + n]})
    assert len({(r["source"], r["start"], r["tokens"]) for r in rows}) == 32
    with open(os.path.join(out, f"prompts-{m}.txt"), "w") as f:
        for r in rows: f.write(" ".join(map(str, r["ids"])) + "\n")
    json.dump([{k: v for k, v in r.items() if k != "ids"} for r in rows], open(os.path.join(out, f"prompts-{m}.json"), "w"), indent=1)
    print(m, "lengths", sorted({r["tokens"] for r in rows}), "sources", [len(x) for x in ids])
