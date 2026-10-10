#!/usr/bin/env python3
"""t3_table.py <data/t3> <out.csv> [token...]: H1 test 3 table from the accepted timing processes
<token>-r<n>.json. Per token and shape: kernel dispatched, the process medians (kernel_median_us), their median,
rate = 2*M*N*K / t (TOP/s), ratio to the incumbent `base` (t_base / t). Per token and model: weighted rate over the
layer's seven linear calls (wq_wo x2, wk_wv x2, w1_w3 x2, w2 x1) and its ratio to base. Pure recomputation from the
JSON files; nothing else is read."""
import csv, glob, json, os, re, statistics as st, sys
D, OUT = sys.argv[1], sys.argv[2]
W = {"wq_wo": 2, "wk_wv": 2, "w1_w3": 2, "w2": 1}; REG = 231.37
runs = {}
for f in sorted(glob.glob(D + "/*-r[0-9]*.json")):
    m = re.match(r"(.*)-r(\d+)\.json$", os.path.basename(f)); tok = m[1]
    if len(sys.argv) > 3 and tok not in sys.argv[3:]: continue
    for c in json.load(open(f))["cases"]:
        if c["suite"] != "linear" or c["scheme"] != "8da4w" or c["regime"] != "prefill": continue
        runs.setdefault(tok, {}).setdefault((c["model"], c["op"]), []).append((int(m[2]), c))
rows = [["token", "model", "op", "M", "N", "K", "kernel", "processes", "process_medians_us", "spread_pct", "kernel_ms", "rate_TOPs", "pct_register_roof", "speed_vs_base"]]
t = {tok: {k: st.median(c["kernel_median_us"] for _, c in v) for k, v in d.items()} for tok, d in runs.items()}
for tok, d in runs.items():
    for (model, op), v in sorted(d.items()):
        c = v[0][1]; med = sorted(x["kernel_median_us"] for _, x in v); ms = st.median(med) / 1000
        rate = 2 * c["M"] * c["N"] * c["K"] / (ms * 1e-3) / 1e12
        kern = ";".join(sorted({x["kernel"].replace("_texture3d_texture2d_half", "") for _, x in v}))
        rows.append([tok, model, op, c["M"], c["N"], c["K"], kern, len(v), " ".join(f"{x:.1f}" for x in med), f"{100 * (med[-1] - med[0]) / st.median(med):.1f}",
                     f"{ms:.4f}", f"{rate:.2f}", f"{100 * rate / REG:.1f}", f"{t['base'][(model, op)] / t[tok][(model, op)]:.3f}" if "base" in t else ""])
    for model in sorted({k[0] for k in d}):
        ops = sum(W[op] * 2 * v[0][1]["M"] * v[0][1]["N"] * v[0][1]["K"] for (mm, op), v in d.items() if mm == model)
        tt = sum(W[op] * t[tok][(mm, op)] for (mm, op) in d if mm == model) * 1e-6
        rate = ops / tt / 1e12
        base = ""
        if "base" in t: base = f"{sum(W[op] * t['base'][(mm, op)] for (mm, op) in d if mm == model) / sum(W[op] * t[tok][(mm, op)] for (mm, op) in d if mm == model):.3f}"
        rows.append([tok, model, "WEIGHTED(2,2,2,1)", "", "", "", "", "", "", "", f"{tt * 1e3:.4f}", f"{rate:.2f}", f"{100 * rate / REG:.1f}", base])
csv.writer(open(OUT, "w", newline="")).writerows(rows)
for r in rows:
    if r[2].startswith("WEIGHTED"): print(*r[:3], r[10], r[11], r[12], r[13], sep="\t")
