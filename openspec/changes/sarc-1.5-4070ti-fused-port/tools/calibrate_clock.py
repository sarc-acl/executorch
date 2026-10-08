#!/usr/bin/env python3
"""calibrate_clock.py <session raw dir> [...] > results/4070ti/clkmin.json

The clock floor of a timed run, from the A/A session (run with --calibrate), by the rule fixed in
tools/thresholds.txt before the session existed: clkmin = floor(clk_factor x the lowest per-run median of
nvidia-smi clocks.gr inside the measured window, over the usable A/A runs), one device-wide value written for
every cell. Usable = rc 0, 2048 prompt tokens, no foreign GPU process, at least clkn clock samples and a per-run
median of at least clk_ramp_mhz (a run that starts while the card still ramps from its idle clock is listed
under ramp_runs and not used). Refused when a cell has fewer than 10 usable runs."""
import csv, hashlib, json, math, os, statistics as st, sys, collections
TH = dict(l.strip().split("=", 1) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "thresholds.txt")) if "=" in l and not l.startswith("#"))
FACTOR, RAMP, CLKN, MIN_RUNS = float(TH["clk_factor"]), float(TH["clk_ramp_mhz"]), int(TH["clkn"]), 10
acc = collections.defaultdict(list); src = []; ramp = []
for d in sys.argv[1:]:
    p = os.path.join(d, "runs.csv"); src.append({"runs_csv": os.path.relpath(p, os.path.join(d, "../../..")), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()})
    for r in csv.DictReader(open(p)):
        if r["log"].startswith("logs/prefill") and r["rc"] == "0" and r["prompt_tokens"] == "2048" and not r["others"] and int(r["clk_n"] or 0) >= CLKN:
            c = f'{r["model"]}-{r["scheme"]}'; v = float(r["clk_med_mhz"])
            (acc[c] if v >= RAMP else ramp).append(v if v >= RAMP else f'{c}:{r["build"]}:r{r["rep"]}:{v}')
names = [f"{m}-{q}" for m in ("1b", "3b", "8b") for q in ("4w", "8da4w")]
bad = [f"{c}: {len(acc.get(c, []))} usable runs, need {MIN_RUNS}" for c in names if len(acc.get(c, [])) < MIN_RUNS]
if bad: sys.exit("calibration refused:\n  " + "\n  ".join(bad))
lowest = min(v for c in names for v in acc[c]); clkmin = math.floor(FACTOR * lowest)
cells = {c: {"clkmin_mhz": clkmin, "cell_median_mhz": st.median(acc[c]), "min_mhz": min(acc[c]), "max_mhz": max(acc[c]), "runs": len(acc[c])} for c in names}
json.dump({"rule": f"floor({FACTOR} * lowest per-run median clocks.gr of the usable A/A runs), device-wide; runs below {RAMP:.0f} MHz are ramp runs",
           "lowest_run_median_mhz": lowest, "ramp_runs": ramp, "sources": src, "cells": cells}, sys.stdout, indent=1); print()
