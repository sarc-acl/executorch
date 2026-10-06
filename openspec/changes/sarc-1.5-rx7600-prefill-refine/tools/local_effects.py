#!/usr/bin/env python3
"""local_effects.py <full-r*.csv>...: effect of each single 4w parameter among the confirmed configurations.

For every pair of confirmed configurations that differ in exactly one parameter (neighbours.py's definition), the
ratio of their kernel times per shape (full measurement, median over the repeats a configuration has). Prints, per
parameter and direction of change, the number of pairs x shapes, the median, the minimum and the maximum ratio.
A ratio above 1 means the second level is slower. Only configurations near the optimum are in these files, so this
is the response surface around the best, not over the whole space (that is importance.py on the random sample)."""
import collections, csv, pathlib, statistics, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import enum_space as es, gen_space_names as gn, neighbours as nb
t = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sys.argv[1:]:
    for r in csv.DictReader(open(f)):
        if r["dispatched"] == "1" and r["us"]: t[r["token"]][(r["model"], r["op"])].append(float(r["us"]))
med = {k: {s: statistics.median(v) for s, v in d.items()} for k, d in t.items()}
def level(c, p):
    if p in ("M", "N", "K", "X", "Y", "S"): return c["g"]["MNKXYS".index(p)]
    if p == "ACC": return next(k for k, on in nb.ACC.items() if all(c["f"].get(x, False) == (x in on) for ks in nb.ACC.values() for x in ks))
    if p == "CSH": return next(k for k, on in nb.CSH.items() if all(c["f"].get(x, False) == (x in on) for ks in nb.CSH.values() for x in ks))
    if p == "TEXEL_STAGING": return "bx" if c["bx"] else "plain"
    return "on" if c["f"].get(p, False) else "off"
eff = collections.defaultdict(list)
for a in med:
    ca = gn.parse_4w(a)
    for b, p in nb.neighbours(a, 1).items():
        if b not in med or a >= b: continue
        cb = gn.parse_4w(b); la, lb = level(ca, p), level(cb, p)
        if str(la) > str(lb): la, lb, x, y = lb, la, b, a
        else: x, y = a, b
        for s in med[x]:
            if s in med[y]: eff[(p, la, lb)].append(med[y][s] / med[x][s])
print("parameter,from,to,pairs_x_shapes,median_ratio,min_ratio,max_ratio")
for (p, la, lb), v in sorted(eff.items(), key=lambda x: (x[0][0], str(x[0][1]), str(x[0][2]))):
    print(f"{p},{la},{lb},{len(v)},{statistics.median(v):.3f},{min(v):.3f},{max(v):.3f}")
