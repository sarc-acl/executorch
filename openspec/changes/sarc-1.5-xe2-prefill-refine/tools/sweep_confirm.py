#!/usr/bin/env python3
"""sweep_confirm.py <space> <checked.csv> <results.csv> <out.csv> [incumbent arm, default base]

The confirmation step of the sampled search: the full measurement (all three models, two repeats, arms
interleaved) of the 10 best configurations per shape class and of the reference arms (sweep_run.py full on the
sw3 build). One row per arm: its parameters and, per (model, shape), the median time relative to the incumbent
arm (x > 1 = faster). Prints, per (model, shape), the best arm, its gain and the repeat spread; a gain inside
+-2 % of the incumbent is not a gain."""
import collections, csv, statistics as st, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from sweep import PARAMS
space, cfgf, resf, out = sys.argv[1:5]; inc = sys.argv[5] if len(sys.argv) > 5 else "base"; LINEAR = space in ("4w", "8da4w")
cfg = {c["id"]: c for c in csv.DictReader(open(cfgf))}; t = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(resf)):
    if r["mode"] != "full" or not r["us"]: continue
    if LINEAR and r["id"] in cfg and r["dispatched"] != "1": continue
    if not LINEAR and r["shape"] != space: continue
    if space == "av" and r["id"] in cfg and int(r["K"]) % int(cfg[r["id"]]["N"]): continue     # the tile does not run this head_dim
    t[r["id"]][(r["model"], r["shape"])].append(float(r["us"]))
shapes = sorted({k for a in t.values() for k in a}); P = PARAMS[space]
med = {a: {k: st.median(v) for k, v in d.items()} for a, d in t.items()}
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["arm"] + P + [f"{m}:{s}_us" for m, s in shapes] + [f"{m}:{s}_x" for m, s in shapes])
    for a in sorted(med, key=lambda a: (a in cfg, a)):
        w.writerow([a] + [cfg.get(a, {}).get(p, "") for p in P] + [f"{med[a][k]:.1f}" if k in med[a] else "" for k in shapes]
                   + [f"{med[inc][k] / med[a][k]:.3f}" if k in med[a] and k in med[inc] else "" for k in shapes])
print(f"{space}: {len(med)} arms, incumbent {inc}")
for k in shapes:
    c = [(med[a][k], a) for a in med if k in med[a] and a in cfg]
    if not c or k not in med[inc]: continue
    us, a = min(c); sp = max((max(v) - min(v)) / st.median(v) * 100 for v in (t[a][k], t[inc][k]))
    refs = " ".join(f"{r}={med[inc][k] / med[r][k]:.3f}" for r in med if r not in cfg and r != inc and k in med[r])
    print(f"  {k[0]} {k[1]}: best {a} {med[inc][k] / us:.3f}x of {inc} ({us:.1f} against {med[inc][k]:.1f} us, repeat spread {sp:.2f} %) {refs}")
