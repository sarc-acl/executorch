#!/usr/bin/env python3
"""session_table.py <runs.csv> [reps=5]: the session table of RESULT.md from a runs.csv of e2e5.sh alone.
Per cell: median tok/s of the first <reps> valid timed runs (log starts with logs/prefill, valid == 1, in file
order) of each arm (build parent = old topic4, cand = new f1), new / old, the spread of each arm (max - min over
its median), the GPU clock (min to max of the per-run medians over all timed runs of the cell) and the invalid runs. Then
the geomean and the pre-registered verdict of LOG.md: "no effect" = every cell inside +-2 % and the geomean
inside +-1 %."""
import csv, math, statistics as st, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["log"].startswith("logs/prefill")]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
cells = {}
for r in rows: cells.setdefault((r["model"], r["scheme"]), []).append(r)
print("| cell | old `topic4` | new `f1` | new / old | difference | spread old / new | valid old / new | invalid | clock MHz |")
print("|---|---:|---:|---:|---:|---|---|---|---|")
ratios = []; complete = True
for (m, q), rs in cells.items():
    v = {b: [float(r["tok_s"]) for r in rs if r["build"] == b and r["valid"] == "1"][:REPS] for b in ("parent", "cand")}
    inv = [f'{r["build"]} r{r["rep"]} {r["reason"]}' for r in rs if r["valid"] != "1"]
    fb = [float(r["clk_med_mhz"]) for r in rs if r["clk_med_mhz"]]
    if min(len(v["parent"]), len(v["cand"])) < REPS:
        complete = False; print(f'| {m.upper()} {q} | INCOMPLETE | | | | | {len(v["parent"])} / {len(v["cand"])} | {"; ".join(inv)} | |'); continue
    a, b = st.median(v["parent"]), st.median(v["cand"]); x = b / a; ratios.append(x)
    sp = lambda l: (max(l) - min(l)) / st.median(l) * 100
    print(f'| {m.upper()} {q} | {a:.2f} | {b:.2f} | {x:.5f} | {(x - 1) * 100:+.2f} % | {sp(v["parent"]):.1f} / {sp(v["cand"]):.1f} % | {len(v["parent"])} / {len(v["cand"])} | {"; ".join(inv) or "none"} | {min(fb):.0f} to {max(fb):.0f} |')
if ratios:
    g = math.exp(sum(map(math.log, ratios)) / len(ratios)); worst = max(ratios, key=lambda x: abs(x - 1))
    ok = complete and len(ratios) == 6 and all(abs(x - 1) <= 0.02 for x in ratios) and abs(g - 1) <= 0.01
    print(f"\nGeomean over {len(ratios)} cells: {g:.5f} ({(g - 1) * 100:+.2f} %); largest cell difference {(worst - 1) * 100:+.2f} %; "
          f"{len(rows)} timed runs, {sum(r['valid'] == '1' for r in rows)} valid. Verdict: {'NO EFFECT (every cell inside +-2 %, geomean inside +-1 %)' if ok else 'EFFECT or incomplete: see the cells'}")
