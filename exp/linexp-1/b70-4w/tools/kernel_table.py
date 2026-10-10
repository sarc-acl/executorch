#!/usr/bin/env python3
"""kernel_table.py <data/kernel> <out.csv> <base config> [config...]: table from the timing processes
<config>/linear-r<n>.json (test_llama_microbench --linear --regime=prefill --storage=texture3d --scheme=4w).
Per config and shape: kernel dispatched, the process medians (kernel_median_us), their median, rate =
2*M*N*K / t (TFLOP/s), share of the register roof (173.3 TFLOP/s, study), ratio to the base config
(t_base / t). Per config and model: weighted rate over the layer's seven linear calls (wq_wo x2, wk_wv x2,
w1_w3 x2, w2 x1) and its ratio to base; GEOMEAN rows: geometric mean of the per-shape rates over all 12 shapes
and over the shapes whose kernel name contains `g82s16m8flib` in the base config. `kind` comes from
<config>/KIND if present (e.g. "ablation, incorrect by construction"); `pdiff` = number of production-diff
logs of that config that say ALL PASSED. Pure recomputation from the files; nothing else is read."""
import csv, glob, json, math, os, re, statistics as st, sys
D, OUT, BASE = sys.argv[1], sys.argv[2], sys.argv[3]
cfgs = [BASE] + [c for c in sys.argv[4:] if c != BASE]
W = {"wq_wo": 2, "wk_wv": 2, "w1_w3": 2, "w2": 1}; REG = 173.3
runs = {}
for cfg in cfgs:
    for f in sorted(x for x in glob.glob(f"{D}/{cfg}/linear-r*.json") if re.search(r"linear-r\d+\.json$", x)):
        for c in json.load(open(f))["cases"]:
            if c["suite"] != "linear" or c["scheme"] != "4w" or c["regime"] != "prefill": continue
            runs.setdefault(cfg, {}).setdefault((c["model"], c["op"]), []).append(c)
t = {cfg: {k: st.median(c["kernel_median_us"] for c in v) for k, v in d.items()} for cfg, d in runs.items()}
flib = {k for k, v in runs[BASE].items() if "g82s16m8flib" in v[0]["kernel"]}
rows = [["config", "kind", "pdiff_passed", "model", "op", "M", "N", "K", "kernel", "processes", "process_medians_us", "spread_pct", "kernel_ms", "rate_TFLOPs", "pct_register_roof", "speed_vs_base"]]
for cfg in cfgs:
    d = runs.get(cfg, {}); kind = open(f"{D}/{cfg}/KIND").read().strip() if os.path.exists(f"{D}/{cfg}/KIND") else "correct kernel"
    pd = sum(1 for f in glob.glob(f"{D}/{cfg}/pdiff-*.log") if "ALL PASSED" in open(f).read() and "FAILED" not in open(f).read())
    rate = {}
    for (model, op), v in sorted(d.items()):
        c = v[0]; med = sorted(x["kernel_median_us"] for x in v); ms = st.median(med) / 1000
        rate[(model, op)] = 2 * c["M"] * c["N"] * c["K"] / (ms * 1e-3) / 1e12
        kern = ";".join(sorted({x["kernel"].replace("_texture3d_texture2d_half", "") for x in v}))
        rows.append([cfg, kind, pd, model, op, c["M"], c["N"], c["K"], kern, len(v), " ".join(f"{x:.1f}" for x in med), f"{100 * (med[-1] - med[0]) / st.median(med):.1f}",
                     f"{ms:.4f}", f"{rate[(model, op)]:.2f}", f"{100 * rate[(model, op)] / REG:.1f}", f"{t[BASE][(model, op)] / t[cfg][(model, op)]:.3f}"])
    for model in sorted({k[0] for k in d}):
        ks = [k for k in d if k[0] == model]
        ops = sum(W[k[1]] * 2 * d[k][0]["M"] * d[k][0]["N"] * d[k][0]["K"] for k in ks); tt = sum(W[k[1]] * t[cfg][k] for k in ks) * 1e-6
        rows.append([cfg, kind, pd, model, "WEIGHTED(2,2,2,1)", "", "", "", "", "", "", "", f"{tt * 1e3:.4f}", f"{ops / tt / 1e12:.2f}", f"{100 * ops / tt / 1e12 / REG:.1f}",
                     f"{sum(W[k[1]] * t[BASE][k] for k in ks) / sum(W[k[1]] * t[cfg][k] for k in ks):.3f}"])
    for name, ks in (("GEOMEAN(12 shapes)", list(d)), ("GEOMEAN(shapes on g82 flib in base)", [k for k in d if k in flib])):
        if ks:
            g = math.exp(st.mean(math.log(rate[k]) for k in ks))
            rows.append([cfg, kind, pd, "all", name, "", "", "", "", len(ks), "", "", "", f"{g:.2f}", f"{100 * g / REG:.1f}",
                         f"{math.exp(st.mean(math.log(t[BASE][k] / t[cfg][k]) for k in ks)):.3f}"])
csv.writer(open(OUT, "w", newline="")).writerows(rows)
for r in rows:
    if r[4].startswith(("WEIGHTED", "GEOMEAN")): print(r[0], r[1][:8], r[3], r[4], r[13], r[14], r[15], sep="\t")
