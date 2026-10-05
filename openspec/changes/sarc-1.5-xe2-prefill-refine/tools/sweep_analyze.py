#!/usr/bin/env python3
"""sweep_analyze.py <space> <out dir> <checked.csv>[,<checked.csv>...] <results.csv>[,<results.csv>...]

Analysis of the sampled parameter search (tools/sweep.py, tools/sweep_run.py) for one space. Needs numpy (run
it with the venv python of the artifact directory, host.sh XE2_PYTHON). Writes into <out dir>:

  validation.csv    the cheap screening mode against the full measurement on the validation configurations:
                    Spearman rank correlation per shape (cheap time of the 1B / 1B+3B shape of a class against
                    the median full time of each model's shape of that class) and for the score
  drift.csv         the `base` arm repeated through the run: spread per shape
  importance.csv    per shape class and for the score: share of the variance of log(time) explained by each
                    parameter alone (eta squared of its main effect), its best level and that level's median
                    time relative to the overall median
  interactions.csv  per parameter pair: variance share explained by the pair jointly minus the share of the
                    additive model of the two main effects (least squares), i.e. the interaction
  ranking.csv       every measured configuration: time per shape, speed relative to the base arm
                    (x > 1 = faster than the device's table kernel), score, correctness where it was checked
  best.csv          the 20 best by score plus the 5 best of each shape class (the refinement seeds)
  top10.csv         the 10 best per shape class (the confirmation list)
  tocheck.txt       ids near the top of any list that have no correctness row yet

Score: linear = per-layer-weighted time of a 1B layer (2 wq/wo + 2 wk/wv + 2 w1/w3 + w2); QK^T = geometric
mean over the head_dim 64 and 128 shapes; attn*V = the shape class itself (64 and 128 are ranked separately,
a tile whose N does not divide head_dim does not run that class). Only shapes the configuration's own kernel
ran are used; a configuration with a failed correctness case is excluded from best.csv and top10.csv, one
whose correctness was not exercised is marked `unverified`.
"""
import collections, csv, itertools, math, os, statistics as st, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import PARAMS

space, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
cfg = {c["id"]: c for p in sys.argv[3].split(",") for c in csv.DictReader(open(p)) if c.get("run", "1") == "1" and c["space"] == space}
rows = [r for p in sys.argv[4].split(",") for r in csv.DictReader(open(p))]
LINEAR = space in ("4w", "8da4w"); P = PARAMS[space]
W = {"wq_wo": 2, "wk_wv": 2, "w1_w3": 2, "w2": 1}
def cls(r):
    """Shape class of a result row, or None if it is not one this space ranks."""
    if LINEAR: return r["shape"] if r["shape"] in W else None
    if r["shape"] != space: return None
    return f'hd{r["K"]}'
CLASSES = list(W) if LINEAR else ["hd64", "hd128"]
def runs(c, k):
    if LINEAR or space == "qk" or c is None: return True
    return int(k[2:]) % int(c["N"]) == 0                                   # attn*V: the tile's N divides head_dim

cheap = collections.defaultdict(dict); full = collections.defaultdict(lambda: collections.defaultdict(list)); corr = {}; basev = collections.defaultdict(list)
for r in rows:
    if r["mode"] == "corr":
        if r["correct"] and "/" in r["correct"]:
            p, t = map(int, r["correct"].split("/")); a = corr.setdefault(r["id"], [0, 0]); a[0] += p; a[1] += t
        continue
    k = cls(r)
    if k is None or not r["us"]: continue
    if LINEAR and r["id"] not in ("base",) and r["dispatched"] != "1": continue
    if r["id"] in cfg and not runs(cfg[r["id"]], k): continue
    if r["mode"] == "cheap":
        if r["id"] == "base": basev[k].append(float(r["us"]))
        else: cheap[r["id"]][k] = float(r["us"])
    elif r["mode"] == "full": full[r["id"]][(r["model"], k)].append(float(r["us"]))
base = {k: st.median(v) for k, v in basev.items()}
def score(t):
    """Lower is better; None if a needed shape is missing."""
    if LINEAR: return sum(W[k] * t[k] for k in W) if all(k in t for k in W) else None
    if space == "qk": return math.sqrt(t["hd64"] * t["hd128"]) if len(t) == 2 else None
    return None
def verdict(i):
    if i not in corr: return ""
    p, t = corr[i]; return "unverified" if t == 0 else ("ok" if p == t else "FAILED")
def spearman(a, b):
    ra, rb = (np.argsort(np.argsort(x)).astype(float) for x in (a, b)); return float(np.corrcoef(ra, rb)[0, 1])

# --- validation: cheap against full ---
with open(f"{out}/validation.csv", "w") as f:
    f.write("class,model,n,spearman_cheap_vs_full,full_rep_spread_median_pct\n")
    ids = [i for i in cheap if i in full and i in cfg]
    models = sorted({m for i in ids for m, _ in full[i]})
    for k in CLASSES:
        for m in models:
            p = [(cheap[i][k], st.median(full[i][(m, k)]), full[i][(m, k)]) for i in ids if k in cheap[i] and (m, k) in full[i]]
            if len(p) >= 5:
                sp = st.median((max(v) - min(v)) / st.median(v) * 100 for _, _, v in p)
                f.write(f"{k},{m},{len(p)},{spearman([x for x, _, _ in p], [y for _, y, _ in p]):.4f},{sp:.2f}\n")
    if LINEAR:                                                               # the score against the layer-weighted full time, per model
        for m in models:
            p = [(score(cheap[i]), sum(W[k] * st.median(full[i][(m, k)]) for k in W)) for i in ids if score(cheap[i]) and all((m, k) in full[i] for k in W)]
            if len(p) >= 5: f.write(f"score,{m},{len(p)},{spearman([x for x, _ in p], [y for _, y in p]):.4f},\n")
with open(f"{out}/drift.csv", "w") as f:
    f.write("class,base_runs,median_us,min_us,max_us,spread_pct\n")
    for k in CLASSES:
        v = basev.get(k, [])
        if v: f.write(f"{k},{len(v)},{st.median(v):.2f},{min(v):.2f},{max(v):.2f},{(max(v) - min(v)) / st.median(v) * 100:.2f}\n")

# --- importance and interactions on log(time) ---
def onehot(ids, p):
    lv = sorted({cfg[i][p] for i in ids}); return np.array([[cfg[i][p] == l for l in lv] for i in ids], float), lv
def r2(X, y):
    X = np.hstack([np.ones((len(y), 1)), X]); b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b
    return 1 - float(e @ e) / float(((y - y.mean()) ** 2).sum())
targets = CLASSES + (["score"] if LINEAR or space == "qk" else [])
imp = []; inter = []
for k in targets:
    ids = [i for i in cheap if i in cfg and (score(cheap[i]) if k == "score" else cheap[i].get(k))]
    if len(ids) < 30: continue
    y = np.log(np.array([score(cheap[i]) if k == "score" else cheap[i][k] for i in ids])); med = float(np.median(y)); main = {}
    for p in P:
        X, lv = onehot(ids, p)
        if len(lv) < 2: continue
        main[p] = r2(X, y); lm = {l: float(np.median(y[X[:, j] == 1])) for j, l in enumerate(lv)}; bl = min(lm, key=lm.get)
        imp.append((k, p, len(ids), main[p], len(lv), bl, math.exp(lm[bl] - med), " ".join(f"{l}:{math.exp(v - med):.2f}" for l, v in lm.items())))
    for p, q in itertools.combinations(main, 2):
        Xp, lp = onehot(ids, p); Xq, lq = onehot(ids, q)
        cell = np.array([[cfg[i][p] == a and cfg[i][q] == b for a in lp for b in lq] for i in ids], float); cell = cell[:, cell.sum(0) > 0]
        add = r2(np.hstack([Xp, Xq]), y); joint = r2(cell, y); inter.append((k, p, q, len(ids), joint, add, joint - add, cell.shape[1]))
with open(f"{out}/importance.csv", "w") as f:
    f.write("target,parameter,n,variance_share,levels,best_level,best_level_median_rel,level_medians_rel\n")
    for r in sorted(imp, key=lambda r: (targets.index(r[0]), -r[3])): f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]},{r[5]},{r[6]:.3f},{r[7]}\n")
with open(f"{out}/interactions.csv", "w") as f:
    f.write("target,parameter_a,parameter_b,n,joint_share,additive_share,interaction_share,cells\n")
    for r in sorted(inter, key=lambda r: (targets.index(r[0]), -r[6])): f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.4f},{r[5]:.4f},{r[6]:.4f},{r[7]}\n")

# --- ranking, the refinement seeds and the confirmation list ---
rank = []
for i, t in cheap.items():
    if i not in cfg and not i.startswith("ref-"): continue
    s = score(t); rank.append((i, t, s))
bs = score(base) if base else None
def line(i, t, s):
    c = cfg.get(i, {}); x = lambda k: f"{base[k] / t[k]:.3f}" if k in t and k in base else ""
    return ([i] + [c.get(p, "") for p in P] + [c.get("shared_bytes", "")] + [f"{t[k]:.1f}" if k in t else "" for k in CLASSES] + [x(k) for k in CLASSES]
            + [f"{bs / s:.3f}" if s and bs else "", verdict(i)])
head = ["id"] + P + ["shared_bytes"] + [f"{k}_us" for k in CLASSES] + [f"{k}_x" for k in CLASSES] + ["score_x", "correctness"]
def dump(name, items):
    with open(f"{out}/{name}", "w", newline="") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(line(*r) for r in items)
by_score = sorted([r for r in rank if r[2]], key=lambda r: r[2])
dump("ranking.csv", by_score + [r for r in rank if not r[2]])
ok = lambda r: r[0] in cfg and verdict(r[0]) != "FAILED"
seeds = [r for r in by_score if ok(r)][:20]; top10 = []
for k in CLASSES:
    bk = sorted([r for r in rank if k in r[1] and ok(r)], key=lambda r: r[1][k])
    seeds += [r for r in bk[:5] if r[0] not in {s[0] for s in seeds}]
    top10 += [r for r in bk[:10] if r[0] not in {s[0] for s in top10}]
    if not by_score: seeds += [r for r in bk[:20] if r[0] not in {s[0] for s in seeds}]      # no score (attn*V): 20 per class
dump("best.csv", seeds); dump("top10.csv", top10)
need = []
for k in CLASSES + ["score"]:
    lst = by_score if k == "score" else sorted([r for r in rank if k in r[1]], key=lambda r: r[1][k])
    need += [r[0] for r in [r for r in lst if r[0] in cfg and verdict(r[0]) != "FAILED"][:30] if r[0] not in corr]
open(f"{out}/tocheck.txt", "w").write("\n".join(dict.fromkeys(need)) + ("\n" if need else ""))
print(f"{space}: {len(cheap)} configurations with cheap rows, {len(full)} with full rows, {len(corr)} with correctness rows; base {base}")
print(f"best score_x {bs / by_score[0][2]:.3f} ({by_score[0][0]})" if by_score and bs else "no score")
for k in CLASSES:
    bk = sorted([r for r in rank if k in r[1]], key=lambda r: r[1][k])
    if bk and k in base: print(f"best {k}: {base[k] / bk[0][1][k]:.3f}x ({bk[0][0]}), {sum(1 for r in bk if r[1][k] < base[k])} of {len(bk)} faster than base")
print(f"to check for correctness: {len(dict.fromkeys(need))}")
