#!/usr/bin/env python3
"""r6_analyze.py <stage>/<raw dir>: the monitored session, recomputed from runs.csv, the per-run logs, .clk and
.mon files. Prints (1) per cell and arm the rows by reason, (2) the medians under the rule as decided (first 5
valid rows) where a cell has them, (3) for the owner's ruling only, the medians of the first 5 rows that are
valid on every predicate except `throttle`, (4) what throttle_status values were sampled and where."""
import csv, json, math, os, statistics as st, sys, collections
d = sys.argv[1]; rows = [r for r in csv.DictReader(open(os.path.join(d, "runs.csv"))) if r["log"].startswith("logs/prefill")]
vals = collections.Counter(); persample = collections.Counter(); chk = 0
for r in rows:
    log = os.path.join(d, r["log"]); obs = None
    for line in open(log, errors="replace"):
        i = line.find("PyTorchObserver")
        if i >= 0: obs = json.loads(line[line.index("{", i):])
    a, b = obs["inference_start_ms"] * 1000, obs["prompt_eval_end_ms"] * 1000
    s = [[int(x) for x in l.split()] for l in open(log[:-4] + ".clk") if len(l.split()) == 7]
    w = [x for x in s if a <= x[0] <= b]; o = 0
    for x in w: o |= x[5]; persample[x[5]] += 1
    mon = [l.split(" ", 2) for l in open(log[:-4] + ".mon") if l.endswith("\n")]
    assert float(r["tok_s"]) == obs["prefill_token_per_sec"] and int(r["clk_n"]) == len(w) and int(r["thr_or"], 16) == o and int(r["mon_n"]) == len(mon), r["log"]
    assert abs(float(r["clk_med_mhz"]) - round(st.median(x[1] for x in w) / 1e6, 1)) < 1e-9
    r["_thr_samples"] = sum(1 for x in w if x[5]); r["_n"] = len(w); r["_tmax"] = max(x[6] for x in w) / 100
    r["_clk_thr"] = [x[1] / 1e6 for x in w if x[5]]; r["_clk_no"] = [x[1] / 1e6 for x in w if not x[5]]
    r["_foreign"] = any(f[1] != "-" for f in mon) or any("/foreign/" in f[2] for f in mon)
    vals[r["thr_or"]] += 1; chk += 1
print(f"rows {len(rows)}, all recomputed from logs/.clk/.mon equal runs.csv: {chk}")
print("thr_or per row:", dict(vals), " per sample in window:", dict(persample))
print("reasons:", dict(collections.Counter(r["reason"] or "valid" for r in rows)))
print("rows with a foreign process or DRM client in any monitor sample:", sum(r["_foreign"] for r in rows),
      "; mon_n", min(int(r["mon_n"]) for r in rows), "to", max(int(r["mon_n"]) for r in rows),
      "; own_engine_ms", min(float(r["own_engine_ms"]) for r in rows), "to", max(float(r["own_engine_ms"]) for r in rows),
      "; others before/after non-empty:", sum(bool(r["others"] or r["others_post"]) for r in rows))
print("clk_n", min(r["_n"] for r in rows), "to", max(r["_n"] for r in rows), "; clk_med", min(float(r["clk_med_mhz"]) for r in rows), "to", max(float(r["clk_med_mhz"]) for r in rows),
      "; lowest sample", min(float(r["clk_min_mhz"]) for r in rows), "; peak temp1", max(int(r["temp_max"]) for r in rows), "C, gpu_metrics", max(r["_tmax"] for r in rows), "C; temp_pre", min(int(r["temp_pre"]) for r in rows), "to", max(int(r["temp_pre"]) for r in rows))
ct = [c for r in rows for c in r["_clk_thr"]]; cn = [c for r in rows for c in r["_clk_no"]]
print(f"clock of samples with status != 0: n={len(ct)} median {st.median(ct):.0f} min {min(ct):.0f} max {max(ct):.0f}; with status 0: n={len(cn)} median {st.median(cn):.0f} min {min(cn):.0f} max {max(cn):.0f}")
cells = collections.OrderedDict()
for r in rows: cells.setdefault((r["model"], r["scheme"]), {"parent": [], "cand": []})[r["build"]].append(r)
sp = lambda l: (max(l) - min(l)) / st.median(l) * 100
for name, ok in (("rule as decided (first 5 valid rows)", lambda r: r["valid"] == "1"),
                 ("NOT the rule, for the owner's ruling: first 5 rows valid on every predicate but `throttle`", lambda r: r["reason"] in ("", "throttle"))):
    print("##", name); ratios = []
    for (m, q), a in cells.items():
        v = {b: [float(r["tok_s"]) for r in a[b] if ok(r)][:5] for b in a}
        n = {b: f'{sum(1 for r in a[b] if r["valid"] == "1")}/{len(a[b])}' for b in a}
        t = {b: f'{sum(r["_thr_samples"] for r in a[b])}/{sum(r["_n"] for r in a[b])}' for b in a}
        if min(len(v["parent"]), len(v["cand"])) < 5: print(f'{m} {q}: INCOMPLETE valid/rows parent {n["parent"]} cand {n["cand"]}; samples with status parent {t["parent"]} cand {t["cand"]}'); continue
        mp, mc = st.median(v["parent"]), st.median(v["cand"]); ratios.append(mc / mp)
        print(f'{m} {q}: {mp:.6g} -> {mc:.6g} {(mc / mp - 1) * 100:+.6f} % spread {sp(v["parent"]):.2f}/{sp(v["cand"]):.2f} valid/rows parent {n["parent"]} cand {n["cand"]}; samples with status parent {t["parent"]} cand {t["cand"]}')
    if ratios: print(f"geomean over {len(ratios)} cells: {(math.exp(sum(map(math.log, ratios)) / len(ratios)) - 1) * 100:+.6f} %")
print("## tok/s of rows with and without status in the window, per cell and arm (median, n)")
for (m, q), a in cells.items():
    for b in a:
        x = [float(r["tok_s"]) for r in a[b] if r["reason"] == "throttle"]; y = [float(r["tok_s"]) for r in a[b] if r["reason"] == ""]
        print(f'{m} {q} {b}: with {st.median(x) if x else "-"} (n={len(x)}), without {st.median(y) if y else "-"} (n={len(y)})')
