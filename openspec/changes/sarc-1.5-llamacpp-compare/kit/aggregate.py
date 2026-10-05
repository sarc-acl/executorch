#!/usr/bin/env python3
"""aggregate.py <device> <session dir>... [--out cells.csv]

Reads runs.csv of one or more sessions of one device and writes one line per (model, arm):
  device,model,arm,kind,n_valid,n_invalid,tok_s_median,tok_s_min,tok_s_max,spread_pct,prefill_ms_median,session
tok_s_median is the median over the valid `prefill` runs (repetition 0 of llama-completion arms is tagged
`discard` and never counted). For a llama-bench arm (`lb`) the samples of its one process are the population.
spread_pct = (max - min) / median * 100. Prints the same table to stdout.
"""
import csv
import os
import statistics as st
import sys

args = sys.argv[1:]
out = None
if "--out" in args:
    i = args.index("--out")
    out = args[i + 1]
    del args[i:i + 2]
device, sessions = args[0], args[1:]
cells = {}
for s in sessions:
    for r in csv.DictReader(open(os.path.join(s, "runs.csv"))):
        if r["tag"] != "prefill":
            continue
        c = cells.setdefault((r["model"], r["arm"]), {"kind": r["kind"], "v": [], "ms": [], "bad": 0, "s": os.path.basename(s.rstrip("/"))})
        if r["valid"] != "1":
            c["bad"] += 1
        elif r["kind"] == "lb":
            c["v"] += [float(x) for x in r["samples"].split("/") if x]
        else:
            c["v"].append(float(r["tok_s"]))
            c["ms"].append(float(r["prefill_ms"]))
head = "device,model,arm,kind,n_valid,n_invalid,tok_s_median,tok_s_min,tok_s_max,spread_pct,prefill_ms_median,session"
lines = [head]
order = {"1b": 0, "3b": 1, "8b": 2}
for (m, a), c in sorted(cells.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[1]["kind"], kv[0][1])):
    v = c["v"]
    if v:
        med = st.median(v)
        row = [f"{med:.2f}", f"{min(v):.2f}", f"{max(v):.2f}", f"{(max(v) - min(v)) / med * 100:.2f}",
               f"{st.median(c['ms']):.2f}" if c["ms"] else ""]
    else:
        row = ["", "", "", "", ""]
    lines.append(",".join([device, m, a, c["kind"], str(len(v)), str(c["bad"])] + row + [c["s"]]))
print("\n".join(lines))
if out:
    open(out, "w").write("\n".join(lines) + "\n")
