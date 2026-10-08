#!/usr/bin/python3
"""calibrate.py <A/A raw dir>: the calibration of proposal.md "Thresholds", from the A/A session's runs.csv and
per-run clock files, by the rules fixed there before the session:
  - clock floor: 97 % of the lowest per-run median clock of the valid timed (prefill) runs;
  - repeats: 7 if any cell shows a repeat spread ((max - min) / median, first 7 valid runs) above 2 % in either arm, else 5;
  - thermal mask: for each temperature bit (32 to 47) seen in the windows of the session's timed runs, per cell: median
    clock of the runs whose window carries the bit against those without it; masked (recorded, not rejected) if every
    such comparison is within 1 %, or, where every run of the cell carries it, if their median clock is within 1 % of
    the median of the other cells' valid timed runs; else it rejects. A bit no timed run shows stays rejecting.
  - foreign-busy ceiling: the highest pre-run gpu_busy_percent (busy_pre_max column) of the valid timed runs, rounded up to the
    next multiple of 5, at least 5; a later run above it is invalid (foreign_busy). Idle foreign use of the display stays visible
    because every row keeps its value.
Validity here ignores the thermal reason (that is what is being decided); every other reason stands.
Prints the analysis and the lines for thresholds.txt."""
import csv, os, statistics as st, sys
from collections import defaultdict

d = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(d, "runs.csv"))))
def ok_but_thermal(r):
    rs = [x for x in r["reason"].split("+") if x and x != "thermal_throttle"]
    return not rs
timed = [r for r in rows if r["log"].startswith("logs/prefill") and ok_but_thermal(r)]
clk = [float(r["clk_med_mhz"]) for r in timed]
floor = 0.97 * min(clk)
print(f"valid timed runs (thermal reason ignored): {len(timed)}; per-run median clock {min(clk):.1f} to {max(clk):.1f} MHz")
print(f"clock floor = 0.97 x {min(clk):.1f} = {floor:.1f} MHz")

cells = defaultdict(lambda: defaultdict(list))
for r in timed:
    cells[(r["model"], r["scheme"])][r["build"]].append(float(r["tok_s"]))
reps = 5
for c, arms in cells.items():
    for b, v in arms.items():
        v = v[:7]
        sp = (max(v) - min(v)) / st.median(v) * 100
        print(f"spread {c[0]} {c[1]} {b}: {sp:.2f} % over {len(v)} runs")
        if sp > 2: reps = 7
print(f"repeats = {reps}")

def bits(r):
    return (int(r["throttle"], 16) >> 32) & 0xFFFF if r["throttle"] else 0
mask = 0xFFFF
allv = timed  # the rule is about timed cells; untimed next-token runs (real, check) carry no timing
for bit in range(16):
    carry = [r for r in allv if bits(r) >> bit & 1]
    if not carry: continue
    ok = True; lines = []
    for key in sorted({(r["model"], r["scheme"], r["log"].split("/")[1].split("-")[0]) for r in carry}):
        same = [r for r in allv if (r["model"], r["scheme"], r["log"].split("/")[1].split("-")[0]) == key]
        w = [float(r["clk_med_mhz"]) for r in same if bits(r) >> bit & 1]
        wo = [float(r["clk_med_mhz"]) for r in same if not bits(r) >> bit & 1]
        if wo:
            dlt = (st.median(w) / st.median(wo) - 1) * 100
            lines.append(f"  {key}: {len(w)} runs with bit {32 + bit}, median clock {st.median(w):.1f}; {len(wo)} without, {st.median(wo):.1f} ({dlt:+.2f} %)")
            ok = ok and abs(dlt) <= 1
        else:
            ref = st.median([float(r["clk_med_mhz"]) for r in timed if (r["model"], r["scheme"]) != key[:2]])
            dlt = (st.median(w) / ref - 1) * 100
            lines.append(f"  {key}: all {len(w)} runs carry bit {32 + bit}, median clock {st.median(w):.1f} vs {ref:.1f} of the other cells' timed runs ({dlt:+.2f} %)")
            ok = ok and abs(dlt) <= 1
    print(f"bit {32 + bit}: {'masked (no clock effect)' if ok else 'REJECTS'}"); print("\n".join(lines))
    if ok: mask &= ~(1 << bit)
print(f"thermal_mask = {mask:#06x}")
busy = [int(r["busy_pre_max"]) for r in timed if r.get("busy_pre_max", "") != ""]
bmax = max(5, -(-max(busy) // 5) * 5) if busy else 100
print(f"pre-run busy % of the valid timed runs: max {max(busy) if busy else None}; ceiling {bmax}")
print(f"\n# thresholds.txt lines\nclkmin={floor:.0f}\nreps={reps}\nthermal_mask={mask:#06x}\nbusy_pre_max={bmax}")
