#!/usr/bin/env python3
"""calibrate_clock.py <session raw dir> [...] > results/4070ti/clkmin.json

The "normal clock" threshold of a timed run, per cell, from the baseline and A/A sessions (run with --calibrate):
normal = median over the usable timed runs of the run's median graphics clock inside the measured window;
clkmin = floor(0.97 * normal). Per cell because a 1B prefill lasts about 100 ms and may sit on a different part
of the clock ramp than an 8B one. Usable = rc 0, expected prompt tokens, no foreign GPU process, at least 2
clock samples. A cell with fewer than 10 usable runs, or whose own runs spread more than 3 % around the median,
is an error: look at the clock samples before measuring candidates."""
import csv, hashlib, json, math, os, statistics as st, sys, collections
FACTOR, MIN_RUNS, MAX_SPREAD = 0.97, 10, 3.0
acc = collections.defaultdict(list); src = []
for d in sys.argv[1:]:
    p = os.path.join(d, "runs.csv"); src.append({"runs_csv": os.path.abspath(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()})
    for r in csv.DictReader(open(p)):
        if r["log"].startswith("logs/prefill") and r["rc"] == "0" and r["prompt_tokens"] == "2048" and not r["others"] and int(r["clk_n"] or 0) >= 2:
            acc[f'{r["model"]}-{r["scheme"]}'].append(float(r["clk_med_mhz"]))
cells, bad = {}, []
for m in ("1b", "3b", "8b"):
    for q in ("4w", "8da4w"):
        v = acc.get(f"{m}-{q}", [])
        if len(v) < MIN_RUNS: bad.append(f"{m}-{q}: {len(v)} usable runs, need {MIN_RUNS}"); continue
        med = st.median(v); spread = (max(v) - min(v)) / med * 100
        if spread > MAX_SPREAD: bad.append(f"{m}-{q}: clock spread {spread:.1f} % (min {min(v)}, max {max(v)})")
        cells[f"{m}-{q}"] = {"clkmin_mhz": math.floor(FACTOR * med), "normal_mhz": med, "min_mhz": min(v), "max_mhz": max(v), "runs": len(v)}
if bad: sys.exit("calibration refused:\n  " + "\n  ".join(bad))
json.dump({"rule": f"floor({FACTOR} * median of per-run median clocks.gr), per cell", "sources": src, "cells": cells}, sys.stdout, indent=1); print()
