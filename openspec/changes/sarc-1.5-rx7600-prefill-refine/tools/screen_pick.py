#!/usr/bin/env python3
"""screen_pick.py <screen.csv> [exclude-substring]: per shape (model, op), the kernels at least 3 % faster than the incumbent ("table") in
EVERY round (R8 screen rule; a tie keeps the incumbent). Only rows with dispatched=1 count. Prints, per shape, the
table time per round, every kernel's per-round speedup (table / kernel), and the pick (the kernel with the best
worst-round speedup among those passing, else the table). Exit 0."""
import collections, csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
excl = sys.argv[2] if len(sys.argv) > 2 else None
if excl: rows = [r for r in rows if excl not in r["cand"]]
t = collections.defaultdict(dict)       # (shape, cand) -> {round: us}
disp = {}
for r in rows:
    k = ((r["model"], r["op"]), r["cand"])
    if r["kernel_median_us"] in ("", "None"): continue
    t[k][int(r["round"])] = float(r["kernel_median_us"]); disp[k] = disp.get(k, 1) and int(r["dispatched"])
shapes = sorted({k[0] for k in t}); cands = sorted({k[1] for k in t} - {"table"})
rounds = sorted({r for k in t for r in t[k]})
print("shape,table_us_by_round,pick,worst_round_speedup")
for s in shapes:
    base = t[(s, "table")]
    best = ("table", 1.0)
    for c in cands:
        k = (s, c)
        if k not in t or not disp.get(k) or any(r not in t[k] or r not in base for r in rounds): continue
        sp = min(base[r] / t[k][r] for r in rounds)
        if sp >= 1.03 and sp > best[1]: best = (c, sp)
    print(f"{s[0]}/{s[1]},{'/'.join(f'{base[r]:.0f}' for r in rounds)},{best[0]},{best[1]:.3f}")
    for c in cands:
        k = (s, c)
        if k in t and disp.get(k) and all(r in t[k] and r in base for r in rounds):
            print(f"   {c},{'/'.join(f'{base[r] / t[k][r]:.3f}' for r in rounds)}", file=sys.stderr)
