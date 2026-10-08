#!/usr/bin/env python3
"""revalidate.py <raw dir> <out dir> [thresholds.txt]: re-judge the runs of a session (runs.csv, left untouched) under the
thresholds of thresholds.txt (clkmin, thermal_mask, busy_pre_max, clkn) and write <out dir>/runs.csv (valid / reason recomputed)
and <out dir>/nexttoken.csv. Used for the A/A session, which was run before its own calibration was written (every temperature
bit rejected, no clock floor), and wherever a session must be judged under later thresholds. Reasons other than the three
threshold-dependent ones (thermal_throttle, clock_low, foreign_busy) are kept as recorded."""
import csv, os, shutil, sys
raw, out = sys.argv[1], sys.argv[2]
th = {}
for l in open(sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "thresholds.txt")):
    if "=" in l and not l.startswith("#"):
        k, v = l.strip().split("=", 1); th[k] = v
clkmin, mask, bmax = float(th["clkmin"]), int(th["thermal_mask"], 16), int(th["busy_pre_max"])
rows = list(csv.DictReader(open(os.path.join(raw, "runs.csv"))))
keep = {"thermal_throttle", "clock_low", "foreign_busy"}
for r in rows:
    rs = [x for x in r["reason"].split("+") if x and x not in keep]
    if r["throttle"] and (int(r["throttle"], 16) >> 32) & mask: rs.append("thermal_throttle")
    if "clock_unsampled" not in rs and r["clk_med_mhz"] and float(r["clk_med_mhz"]) < clkmin: rs.append("clock_low")
    if r.get("busy_pre_max", "") != "" and int(r["busy_pre_max"]) > bmax: rs.append("foreign_busy")
    r["reason"] = "+".join(rs); r["valid"] = "0" if rs else "1"
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "runs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
if os.path.exists(os.path.join(raw, "nexttoken.csv")): shutil.copy(os.path.join(raw, "nexttoken.csv"), out)
print(f"{sum(r['valid'] == '1' for r in rows)} of {len(rows)} runs valid under {th}")
