#!/usr/bin/env python3
"""validate_screen.py <full.csv> <mode.csv>...: does a cheap screening mode rank 4w configurations like the full
measurement (12 shapes, 3 warm-up + 5 timed runs)? For every model: Spearman rank correlation between the mode's
kernel time and (a) the same shape in the full measurement, (b) the model's linear time per layer from the full
measurement (2 wq_wo + 2 wk_wv + 2 w1_w3 + w2), (c) every other shape; and how many of the full top 10 are in the
mode's top 10 / top 20. Only configurations that dispatched their own kernel on the shapes compared are ranked."""
import csv, collections, sys
W = {"wq_wo": 2, "wk_wv": 2, "w1_w3": 2, "w2": 1}
def load(p):
    d = collections.defaultdict(dict)
    for r in csv.DictReader(open(p)):
        if r["us"] and r["dispatched"] == "1": d[r["token"]][(r["model"], r["op"])] = float(r["us"])
    return d
def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2
        i = j + 1
    return r
def spearman(a, b):
    ra, rb = ranks(a), ranks(b); n = len(a); ma, mb = sum(ra) / n, sum(rb) / n
    c = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)); va = sum((x - ma) ** 2 for x in ra); vb = sum((y - mb) ** 2 for y in rb)
    return c / (va * vb) ** 0.5 if va and vb else float("nan")
full = load(sys.argv[1]); models = sorted({m for d in full.values() for m, _ in d})
print("mode,model,n,rho_same_shape,rho_model_layer_time,top10_in_top10,top10_in_top20," + ",".join("rho_" + o for o in W))
for p in sys.argv[2:]:
    mode = load(p); name = p.split("/")[-1].replace(".csv", "")
    for m in models:
        ops = sorted({o for d in mode.values() for mm, o in d if mm == m})
        if not ops: continue
        op = ops[0]
        toks = [t for t in mode if (m, op) in mode[t] and all((m, o) in full.get(t, {}) for o in W)]
        x = [mode[t][(m, op)] for t in toks]; layer = [sum(W[o] * full[t][(m, o)] for o in W) for t in toks]
        top = lambda v, k: {toks[i] for i in sorted(range(len(v)), key=lambda i: v[i])[:k]}
        print(f"{name},{m},{len(toks)},{spearman(x, [full[t][(m, op)] for t in toks]):.3f},{spearman(x, layer):.3f},"
              f"{len(top(layer, 10) & top(x, 10))},{len(top(layer, 10) & top(x, 20))},"
              + ",".join(f"{spearman(x, [full[t][(m, o)] for t in toks]):.3f}" for o in W))
