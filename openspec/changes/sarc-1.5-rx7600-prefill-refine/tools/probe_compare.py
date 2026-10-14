#!/usr/bin/env python3
"""probe_compare.py <probe dir> [out csv]: the real-text comparison of the owner decisions of 2026-10-04.

For every cell (model x scheme) and for three pairs of arms, over the 32 prompts of probe_prompts.py:
top-1 differences, mean and maximum KL divergence of the next-token distribution (nat, KL(first || second)),
maximum absolute logit difference, and the perplexity of the true next token of each arm with their ratio.
Pairs: cand-default vs parent-default (the candidate), parent-tiled vs parent-default (two arms already accepted
as correct on this device, for scale), cand-tiled vs cand-default.
Gross-divergence check (second decision, item 3; a bug detector, thresholds fixed by the decision): a cell fails
when the mean KL of cand-default against parent-default exceeds 0.5 nat or the top-1 token differs on more than
one third of the prompts."""
import json, os, sys
import numpy as np

d = sys.argv[1]; V = 128256
def load(name):
    a = np.fromfile(os.path.join(d, name + ".bin"), dtype=np.float32)
    assert a.size == 32 * V, (name, a.size)
    return a.reshape(32, V).astype(np.float64)
def logp(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))
rows = ["model,scheme,pair,prompts,top1_diff,kl_mean_nat,kl_max_nat,max_abs_logit_diff,ppl_first,ppl_second,ppl_ratio,gross_divergence"]
fail = False
for m in ("1b", "3b", "8b"):
    nxt = np.array([r["next_token"] for r in json.load(open(os.path.join(d, f"prompts-{m}.json")))])
    for q in ("4w", "8da4w"):
        if not os.path.exists(os.path.join(d, f"{m}-{q}-cand-default.bin")): continue
        L = {a: load(f"{m}-{q}-{a}") for a in ("parent-default", "parent-tiled", "cand-default", "cand-tiled")}
        for first, second in (("cand-default", "parent-default"), ("parent-tiled", "parent-default"), ("cand-tiled", "cand-default")):
            a, b = L[first], L[second]; la, lb = logp(a), logp(b)
            kl = (np.exp(la) * (la - lb)).sum(axis=1)
            top = int((a.argmax(axis=1) != b.argmax(axis=1)).sum())
            ppl = [float(np.exp(-x[np.arange(32), nxt].mean())) for x in (la, lb)]
            gross = ""
            if first == "cand-default":
                bad = kl.mean() > 0.5 or top > 32 / 3
                gross = "FAIL" if bad else "ok"; fail = fail or bad
            rows.append(f"{m},{q},{first} vs {second},32,{top},{kl.mean():.3e},{kl.max():.3e},{np.abs(a - b).max():.4f},"
                        f"{ppl[0]:.4f},{ppl[1]:.4f},{ppl[0] / ppl[1]:.5f},{gross}")
text = "\n".join(rows) + "\n"
if len(sys.argv) > 2: open(sys.argv[2], "w").write(text)
sys.stdout.write(text)
print("gross divergence check:", "FAIL" if fail else "ok")
