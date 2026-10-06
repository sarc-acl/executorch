#!/usr/bin/python3
"""fused_pick.py <fused-screen.csv> <incumbent pair>: the screen rule of R8 (fixed in proposal.md) applied per head_dim:
a variant replaces the incumbent (the 780M's choice) only if it is at least 3 % faster in every round on every model
of that head_dim (d64: 1B; d128: 3B and 8B); among several, the fastest median wins; a tie keeps the incumbent.
A pair is "<d64>+<d128>". A variant that occurs in more than one pair is judged by its slowest occurrence in a round. Prints the analysis to
stderr and the chosen pair (ET_VK_SARC_780M_SDPA_FUSED) to stdout."""
import csv, statistics as st, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1]))); inc = sys.argv[2].split(",")
HD = {"d64": ("1b",), "d128": ("3b", "8b")}
t = defaultdict(list)  # (variant, model, round) -> [us]
for r in rows:
    d64, d128 = r["pair"].split("+")
    m = next(k for k in ("1b", "3b", "8b") if k in r["model"].lower())
    v = d64 if m == "1b" else d128
    t[(v, m, r["round"])].append(float(r["median_us"]))
rounds = sorted({r["round"] for r in rows})
pick = []
for i, (hd, models) in enumerate(HD.items()):
    best, best_med = inc[i], None
    variants = sorted({k[0] for k in t if k[1] in models})
    for v in variants:
        rat = [max(t[(v, m, rd)]) / max(t[(inc[i], m, rd)]) for m in models for rd in rounds if (v, m, rd) in t and (inc[i], m, rd) in t]
        med = st.median(max(t[(v, m, rd)]) for m in models for rd in rounds if (v, m, rd) in t)
        ok = v != inc[i] and len(rat) == len(models) * len(rounds) and max(rat) <= 0.97
        print(f"{hd} {v}: vs incumbent per (model, round) {' '.join(f'{x:.3f}' for x in rat)}; median {med:.0f} us; {'QUALIFIES' if ok else ('incumbent' if v == inc[i] else 'no')}", file=sys.stderr)
        if ok and (best_med is None or med < best_med): best, best_med = v, med
    pick.append(best)
print(",".join(pick))
