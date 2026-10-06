#!/usr/bin/env python3
"""load_report.py <session raw dir>: was a run's model load slow? (owner decision 2026-10-06: record it per run.)

From the 20 ms clock samples of each run in runs.csv (logs/<run>.clk: epoch_us clock_MHz ...). A clean process
drives the GPU from its first dispatch to its exit; a process that reads its model from the share leaves the GPU
idle in the middle and the clock falls. gap_s is the time the clock was below 1500 MHz between the first and the
last sample at or above 2000 MHz; a load is called slow at gap_s >= 1.0. Prints one csv row per run:
log,build,rc,duration_s,gap_s,slow,cached_before_pct,warm_passes (the last two from warm.csv when the session
has one), then one summary line per arm. Reports only; no verdict depends on it.
"""
import collections, csv, os, sys

def gap(path):
    rows = [(int(a), float(b)) for a, b, *_ in (l.split() for l in open(path) if len(l.split()) >= 2)]
    if len(rows) < 2: return None, None
    hi = [i for i, (_, c) in enumerate(rows) if c >= 2000]
    g = sum(rows[i + 1][0] - rows[i][0] for i in range(hi[0], hi[-1]) if rows[i][1] < 1500) / 1e6 if hi else 0.0
    return (rows[-1][0] - rows[0][0]) / 1e6, g

def main(d):
    warm = {}
    if os.path.exists(os.path.join(d, "warm.csv")):
        for r in csv.reader(open(os.path.join(d, "warm.csv"))):
            if len(r) >= 6: warm[r[5]] = (r[2], r[3])
    n = collections.Counter(); print("log,build,rc,duration_s,gap_s,slow,cached_before_pct,warm_passes")
    for r in csv.DictReader(open(os.path.join(d, "runs.csv"))):
        clk = os.path.join(d, r["log"][:-4] + ".clk")
        dur, g = gap(clk) if os.path.exists(clk) else (None, None)
        slow = "" if g is None else ("yes" if g >= 1.0 else "no")
        w = warm.get(r["log"], ("", ""))
        print(f'{r["log"]},{r["build"]},{r["rc"]},{"" if dur is None else f"{dur:.1f}"},{"" if g is None else f"{g:.1f}"},{slow},{w[0]},{w[1]}')
        n[(r["build"], slow or "unsampled", r["rc"] != "0")] += 1
    for (b, s, bad), c in sorted(n.items()):
        print(f"# {b}: slow={s} {'non-zero rc' if bad else 'rc 0'}: {c} runs")

if __name__ == "__main__":
    main(sys.argv[1])
