#!/usr/bin/env python3
"""calibrate_clock.py <session raw dir> [...] > results/4070ti/clkmin.json

The "normal clock" threshold of a timed run, from the baseline and A/A sessions (run with --calibrate).
Measured on this card (session s1-aa): the per-run median of nvidia-smi clocks.gr inside the measured window
sits between 2565 and 2790 MHz, differs between cells and between repeats of one cell by up to 5 % with no
effect on the rate, and the first run after an idle period can still show the ramp from the 210 MHz idle clock
(675 MHz median, same rate). So one device-wide threshold is used: normal = the lowest per-cell median of the
per-run median clocks; clkmin = floor(0.97 * normal), written for every cell. Usable = rc 0, expected prompt
tokens, no foreign GPU process, at least 2 clock samples. Refused when a cell has fewer than 10 usable runs or
when more than 10 % of the usable runs fall below the threshold; runs below it are listed."""
import csv, hashlib, json, math, os, statistics as st, sys, collections
FACTOR, MIN_RUNS, MAX_LOW = 0.97, 10, 0.10
acc = collections.defaultdict(list); src = []
for d in sys.argv[1:]:
    p = os.path.join(d, "runs.csv"); src.append({"runs_csv": os.path.abspath(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()})
    for r in csv.DictReader(open(p)):
        if r["log"].startswith("logs/prefill") and r["rc"] == "0" and r["prompt_tokens"] == "2048" and not r["others"] and int(r["clk_n"] or 0) >= 2:
            acc[f'{r["model"]}-{r["scheme"]}'].append(float(r["clk_med_mhz"]))
cells, bad = {}, []
names = [f"{m}-{q}" for m in ("1b", "3b", "8b") for q in ("4w", "8da4w")]
for c in names:
    if len(acc.get(c, [])) < MIN_RUNS: bad.append(f"{c}: {len(acc.get(c, []))} usable runs, need {MIN_RUNS}")
if bad: sys.exit("calibration refused:\n  " + "\n  ".join(bad))
normal = min(st.median(acc[c]) for c in names); clkmin = math.floor(FACTOR * normal)
allv = [v for c in names for v in acc[c]]; low = sorted(f"{c}:{v}" for c in names for v in acc[c] if v < clkmin)
if len(low) > MAX_LOW * len(allv): sys.exit(f"calibration refused: {len(low)} of {len(allv)} usable runs below {clkmin} MHz: {low}")
for c in names:
    v = acc[c]; cells[c] = {"clkmin_mhz": clkmin, "cell_median_mhz": st.median(v), "min_mhz": min(v), "max_mhz": max(v), "runs": len(v)}
json.dump({"rule": f"floor({FACTOR} * lowest per-cell median of per-run median clocks.gr), device-wide", "normal_mhz": normal,
           "runs_below_threshold": low, "sources": src, "cells": cells}, sys.stdout, indent=1); print()
