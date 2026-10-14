#!/usr/bin/env python3
"""r6_analyze.py <stage>/<raw dir> [per-run csv]: a monitored session recomputed from runs.csv, the per-run logs,
.clk and .mon files, under the owner decision of 2026-10-09 04:55 UTC: a run is rejected for throttle only if a
window sample of throttle_status has one of bits 4, 5, 6, 9, 10 or any bit above 12 (mask 0xFFFFE670); the other
bits are counted per run. runs.csv itself is not rewritten (its valid / reason columns are those of the tool
that recorded it); every other predicate is taken as recorded there after being recomputed here. The numbers
are the medians of the first 5 valid rows per arm per cell."""
import csv, json, math, os, statistics as st, sys, collections
REJECT = 0xFFFFE670
d = sys.argv[1]; rows = [r for r in csv.DictReader(open(os.path.join(d, "runs.csv"))) if r["log"].startswith("logs/prefill")]
persample = collections.Counter(); asrec = collections.Counter(); order = 0
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
    foreign = sorted({p for f in mon if f[1] != "-" for p in f[1].split(";")} | {c for f in mon for c in f[2].split(";") if "/foreign/" in c})
    cm = round(st.median(x[1] for x in w) / 1e6, 1)
    assert float(r["tok_s"]) == obs["prefill_token_per_sec"] and int(r["clk_n"]) == len(w) and int(r["thr_or"], 16) == o and int(r["mon_n"]) == len(mon) and float(r["clk_med_mhz"]) == cm, r["log"]
    reason = []
    if r["rc"] != "0": reason.append("rc")
    if obs.get("prompt_tokens") != 2048: reason.append("prompt_tokens")
    if obs.get("generated_tokens") != 0: reason.append("generated_tokens")
    if r["others"] or r["others_post"]: reason.append("other_gpu_process")
    if len(w) < 5: reason.append("clock_unsampled")
    elif cm < 2700: reason.append("clock_low")
    if sum(1 for x in w if x[5] >= 0) < 5: reason.append("throttle_unsampled")
    elif o & REJECT: reason.append("throttle")
    if not mon: reason.append("monitor_unsampled")
    if foreign: reason.append("foreign")
    old = set(r["reason"].split("+")) - {""}
    assert old - {"throttle"} == set(reason) - {"throttle"} or (old & {"other_gpu_process_during", "foreign_gpu_client"} and "foreign" in reason), (r["log"], old, reason)
    r["_valid"] = not reason; r["_reason"] = "+".join(reason); r["_n"] = len(w); r["_tmax"] = max(x[6] for x in w) / 100
    r["_bits"] = ";".join(f"{i}:{c}" for i in range(32) if (c := sum(1 for x in w if x[5] >> i & 1)))
    r["_cmin"] = min(x[1] for x in w) / 1e6; r["_cm"] = cm
    r["_clk_thr"] = [x[1] / 1e6 for x in w if x[5]]; r["_clk_no"] = [x[1] / 1e6 for x in w if not x[5]]
    asrec[r["reason"] or "valid"] += 1
# interleaving: within a cell, rows alternate parent,cand on odd repeats and cand,parent on even ones
cells = collections.OrderedDict()
for r in rows: cells.setdefault((r["model"], r["scheme"]), []).append(r)
viol = sum(1 for c in cells.values() for i in range(0, len(c) - 1, 2)
           if [c[i]["build"], c[i + 1]["build"]] != (["parent", "cand"] if int(c[i]["rep"]) % 2 else ["cand", "parent"]) or c[i]["rep"] != c[i + 1]["rep"])
used = set()
sp = lambda l: (max(l) - min(l)) / st.median(l) * 100
print(f"rows {len(rows)}, recomputed from logs/.clk/.mon and equal to runs.csv in every field checked: {len(rows)}; interleave violations: {viol}")
print("as recorded by the tool at the time (any non-zero status rejected):", dict(asrec))
print("under the decision of 2026-10-09 04:55 UTC (mask 0x%08X):" % REJECT, dict(collections.Counter(r["_reason"] or "valid" for r in rows)))
print("throttle_status per window sample:", {f"0x{k:08x}": v for k, v in sorted(persample.items())})
print("clk_n", min(r["_n"] for r in rows), "to", max(r["_n"] for r in rows), "; clk_med", min(r["_cm"] for r in rows), "to", max(r["_cm"] for r in rows),
      "; lowest sample", min(r["_cmin"] for r in rows), "; peak temp1", max(int(r["temp_max"]) for r in rows), "C, gpu_metrics", max(r["_tmax"] for r in rows),
      "C; temp_pre", min(int(r["temp_pre"]) for r in rows), "to", max(int(r["temp_pre"]) for r in rows),
      "; mon_n", min(int(r["mon_n"]) for r in rows), "to", max(int(r["mon_n"]) for r in rows),
      "; own_engine_ms", min(float(r["own_engine_ms"]) for r in rows), "to", max(float(r["own_engine_ms"]) for r in rows),
      "; load_ms", min(int(r["load_ms"]) for r in rows), "to", max(int(r["load_ms"]) for r in rows), "; resident_pct", min(float(r["resident_pct"]) for r in rows), "to", max(float(r["resident_pct"]) for r in rows))
ct = [c for r in rows for c in r["_clk_thr"]]; cn = [c for r in rows for c in r["_clk_no"]]
if ct: print(f"clock of samples with status != 0: n={len(ct)} median {st.median(ct):.0f} min {min(ct):.0f} max {max(ct):.0f}; with status 0: n={len(cn)} median {st.median(cn):.0f} min {min(cn):.0f} max {max(cn):.0f}")
print("## medians of the first 5 valid rows per arm per cell")
print("cell,parent_med,cand_med,gain_pct,parent_spread_pct,cand_spread_pct,valid/rows parent,valid/rows cand,rows used with a recorded bit parent,cand,window samples with bit 1 parent,cand")
ratios = []
for (m, q), c in cells.items():
    a = {b: [r for r in c if r["build"] == b] for b in ("parent", "cand")}
    u = {b: [r for r in a[b] if r["_valid"]][:5] for b in a}
    for b in u: used.update(r["log"] for r in u[b])
    n = {b: f'{sum(r["_valid"] for r in a[b])}/{len(a[b])}' for b in a}
    t = {b: f'{sum(len(r["_clk_thr"]) for r in a[b])}/{sum(r["_n"] for r in a[b])}' for b in a}
    wb = {b: sum(1 for r in u[b] if r["_bits"]) for b in u}
    if min(len(u["parent"]), len(u["cand"])) < 5: print(f"{m} {q},INCOMPLETE,{n['parent']},{n['cand']}"); continue
    v = {b: [float(r["tok_s"]) for r in u[b]] for b in u}
    mp, mc = st.median(v["parent"]), st.median(v["cand"]); ratios.append(mc / mp)
    print(f'{m} {q},{mp:.6g},{mc:.6g},{(mc / mp - 1) * 100:+.6f},{sp(v["parent"]):.2f},{sp(v["cand"]):.2f},{n["parent"]},{n["cand"]},{wb["parent"]}/5,{wb["cand"]}/5,{t["parent"]},{t["cand"]}')
if ratios: print(f"geomean over {len(ratios)} cells: {(math.exp(sum(map(math.log, ratios)) / len(ratios)) - 1) * 100:+.6f} %")
print("## tok/s of rows with and without a recorded bit in the window, per cell and arm (median, n; all rows)")
for (m, q), c in cells.items():
    for b in ("parent", "cand"):
        x = [float(r["tok_s"]) for r in c if r["build"] == b and r["_bits"]]; y = [float(r["tok_s"]) for r in c if r["build"] == b and not r["_bits"]]
        print(f'{m} {q} {b}: with {st.median(x) if x else "-"} (n={len(x)}), without {st.median(y) if y else "-"} (n={len(y)})')
if len(sys.argv) > 2:
    with open(sys.argv[2], "w") as f:
        f.write("model,scheme,build,rep,slot,tok_s,clk_n,clk_med_mhz,clk_min_mhz,temp_max,thr_or,thr_bits,valid_as_recorded,reason_as_recorded,valid,reason,used_in_median\n")
        for r in rows:
            f.write(",".join(str(x) for x in [r["model"], r["scheme"], r["build"], r["rep"], r["slot"], r["tok_s"], r["_n"], r["_cm"], r["_cmin"], r["temp_max"], r["thr_or"], r["_bits"],
                                               r["valid"], r["reason"], int(r["_valid"]), r["_reason"], int(r["log"] in used)]) + "\n")
