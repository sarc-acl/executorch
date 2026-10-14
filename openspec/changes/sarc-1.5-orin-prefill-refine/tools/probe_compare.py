#!/usr/bin/env python3
"""probe_compare.py <probe/broad dir> <parent default arm> <parent tiled arm> <candidate default arm> [<candidate tiled arm>]

Near-tie evidence, items 2 to 4 of the owner decision of 2026-10-04. For every cell (model x scheme) and the 41
prompts of probe_prompts.py, from the full last-position logits written by logits_dump:
  floor      parent tiled      against parent default   (two arms already accepted as correct on this device)
  candidate  candidate default against parent default   (what the candidate runs against what the parent runs)
  tiled      candidate tiled   against parent tiled     (the reference arm of the gate item, reported only)
Per comparison: prompts with a different top-1 token, mean and maximum KL(reference || other) of the next-token
distribution (nats), maximum absolute logit difference over the vocabulary, and the perplexity of both arms over
the 40 prompts whose next token is known from the text (prompt 0 is a whole file).
Acceptance, per cell and for every metric: candidate <= 2 x floor, i.e.
  top-1 differences, mean KL, max KL, max |logit difference|, and |ln(ppl other / ppl reference)|.
Exit status 0 when every cell passes, 1 otherwise. Prints CSV and a verdict line. Needs numpy."""
import csv, os, sys
import numpy as np
d = sys.argv[1]; P, PT, C = sys.argv[2:5]; CT = sys.argv[5] if len(sys.argv) > 5 else None
meta = list(csv.DictReader(open(os.path.join(d, "prompts_meta.csv")))); N = len(meta)
nxt = np.array([int(r["next_token_in_text"]) for r in meta])
def load(arm, cell):
    a = np.fromfile(os.path.join(d, arm, cell + ".bin"), dtype=np.float32)
    assert a.size % N == 0 and a.size > 0, (arm, cell, a.size); return a.reshape(N, -1).astype(np.float64)
def logp(x): x = x - x.max(axis=1, keepdims=True); return x - np.log(np.exp(x).sum(axis=1, keepdims=True))
def ppl(lp): return float(np.exp(-lp[np.arange(1, N), nxt[1:]].mean()))
def cmp(ref, oth):
    lr, lo = logp(ref), logp(oth); kl = (np.exp(lr) * (lr - lo)).sum(axis=1)
    return dict(top1_diff=int((ref.argmax(1) != oth.argmax(1)).sum()), kl_mean=float(kl.mean()), kl_max=float(kl.max()),
                dlogit_max=float(np.abs(ref - oth).max()), ppl_ref=ppl(lr), ppl_oth=ppl(lo), dppl=abs(float(np.log(ppl(lo) / ppl(lr)))),
                differing=";".join(str(i) for i in np.nonzero(ref.argmax(1) != oth.argmax(1))[0]))
K = ("top1_diff", "kl_mean", "kl_max", "dlogit_max", "dppl")
print("cell,comparison,prompts,top1_diff,kl_mean,kl_max,max_abs_logit_diff,ppl_reference,ppl_other,abs_ln_ppl_ratio,prompts_with_different_top1,within_2x_floor")
ok_all = True
for m in ("1b", "3b", "8b"):
    for q in ("4w", "8da4w"):
        cell = f"{m}-{q}"; p = load(P, cell); fl = cmp(p, load(PT, cell)); ca = cmp(p, load(C, cell))
        fails = [k for k in K if ca[k] > 2 * fl[k]]; ok_all &= not fails
        rows = [("floor: parent tiled vs parent default", fl, ""), ("candidate default vs parent default", ca, "yes" if not fails else "NO: " + "+".join(fails))]
        if CT: rows.append(("candidate tiled vs parent tiled (reported only)", cmp(load(PT, cell), load(CT, cell)), ""))
        for name, r, v in rows:
            print(f'{cell},{name},{N},{r["top1_diff"]},{r["kl_mean"]:.6g},{r["kl_max"]:.6g},{r["dlogit_max"]:.4f},{r["ppl_ref"]:.4f},{r["ppl_oth"]:.4f},{r["dppl"]:.6g},{r["differing"]},{v}')
print(f"verdict: {'WITHIN' if ok_all else 'OUTSIDE'} twice the noise floor in every cell and metric" if ok_all else "verdict: OUTSIDE twice the noise floor (see the NO rows)")
sys.exit(0 if ok_all else 1)
