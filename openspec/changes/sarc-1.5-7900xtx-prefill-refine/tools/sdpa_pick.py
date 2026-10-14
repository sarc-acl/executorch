#!/usr/bin/env python3
"""sdpa_pick.py <sdpa-screen.csv>: the screen rule of R8 (3 % in every round) per op and head dimension. A profile is a QK^T
candidate (name starts with qk) or an attn*V candidate (av); its op time is compared with the table's in every round on every model of the head
dimension (d64: llama-3.2-1b; d128: 3B and 8B); it qualifies if at least 3 % faster in every round on every model; among several the fastest
median wins; a tie keeps the table. Prints the analysis (stderr) and "<hd> <op> <profile|table> <worst-round speedup>" per choice (stdout)."""
import csv, statistics as st, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
t = defaultdict(dict)  # (profile, model, round) -> row
for r in rows: t[(r["profile"], r["model"], int(r["round"]))] = r
profiles = sorted({r["profile"] for r in rows} - {"table"}); rounds = sorted({int(r["round"]) for r in rows})
HD = {"d64": ["llama-3.2-1b"], "d128": ["llama-3.2-3b", "llama-3.1-8b"]}
for hd, models in HD.items():
    for op, pre in (("qk", ("qk",)), ("av", ("av",))):
        best, bsp = "table", 1.0
        for p in profiles:
            if not p.startswith(pre): continue
            sp = []
            for m in models:
                for rd in rounds:
                    a, b = t.get(("table", m, rd)), t.get((p, m, rd))
                    if a and b and a[op + "_us"] not in ("", "None") and b[op + "_us"] not in ("", "None"): sp.append(float(a[op + "_us"]) / float(b[op + "_us"]))
            if len(sp) < len(models) * len(rounds): continue
            ok = min(sp) >= 1.03
            print(f"{hd} {op} {p}: per (model, round) speedup {' '.join(f'{x:.3f}' for x in sp)}{'  QUALIFIES' if ok else ''}", file=sys.stderr)
            if ok and min(sp) > bsp: best, bsp = p, min(sp)
        print(f"{hd} {op} {best} {bsp:.3f}")
