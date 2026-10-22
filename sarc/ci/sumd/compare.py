#!/usr/bin/env python3
"""Summarize e2e-adb.sh runs into results.json and gate them against a baseline.

usage: compare.py --out results.json [--info run-info.txt] [--baseline results.json]
                  [--cell-min 0.95] [--geomean-min 0.97] [--min-valid 3] RUN_DIR...

Each RUN_DIR holds an e2e-adb.sh runs.csv (and nexttoken.csv when the check ran). A cell is
(build, model, scheme); its value is the median prefill tok/s of the valid timed runs (rc 0, a
tok/s value, slot 1/2 or the retry slot 9). Prints TeamCity service messages for every cell.

With --baseline, a build fails when one of its cells is below cell-min times the baseline, or
when the geomean of its ratios is below geomean-min. A next-token check that was SAME in the
baseline and is DIFFER now also fails.

Exit status: 0 pass, 1 regression, 2 bad input (missing cells, too few valid runs).
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path


def tc_escape(s):
    return (
        str(s)
        .replace("|", "||")
        .replace("'", "|'")
        .replace("\n", "|n")
        .replace("[", "|[")
        .replace("]", "|]")
    )


def load(run_dirs, min_valid):
    times = {}
    checks = {}
    for d in run_dirs:
        with open(Path(d) / "runs.csv", newline="") as f:
            for row in csv.DictReader(f):
                if row["slot"] not in ("1", "2", "9"):
                    continue
                key = f"{row['build']}.{row['model']}.{row['scheme']}"
                times.setdefault(key, [])
                if row["rc"] == "0" and row["tok_s"]:
                    times[key].append(float(row["tok_s"]))
        nt = Path(d) / "nexttoken.csv"
        if nt.exists():
            for line in nt.read_text().split():
                m, q, r = line.split(",")
                checks[f"{m}.{q}"] = r
    cells = {}
    bad = []
    for key, v in sorted(times.items()):
        if len(v) < min_valid:
            bad.append(f"{key}: {len(v)} valid runs (need {min_valid})")
            continue
        cells[key] = {"median": statistics.median(v), "runs": v}
    return cells, checks, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--info")
    ap.add_argument("--baseline")
    ap.add_argument("--cell-min", type=float, default=0.95)
    ap.add_argument("--geomean-min", type=float, default=0.97)
    ap.add_argument("--min-valid", type=int, default=3)
    a = ap.parse_args()

    cells, checks, bad = load(a.runs, a.min_valid)
    info = {}
    if a.info:
        for line in Path(a.info).read_text().splitlines():
            k, _, v = line.partition("=")
            info[k] = v
    result: dict = {"info": info, "cells": cells, "nexttoken": checks}

    for key, c in cells.items():
        print(
            f"##teamcity[buildStatisticValue key='sumd.{tc_escape(key)}.tok_s'"
            f" value='{c['median']:.2f}']"
        )

    status = 0
    if bad:
        for b in bad:
            print(f"BAD INPUT {b}")
        status = 2

    if a.baseline:
        base = json.loads(Path(a.baseline).read_text())
        result["baseline"] = {"file": a.baseline, "info": base.get("info", {})}
        ratios = {}
        for key, bc in base["cells"].items():
            if key not in cells:
                print(f"BAD INPUT {key}: in the baseline, not in this run")
                status = 2
                continue
            ratios[key] = cells[key]["median"] / bc["median"]
        result["ratios"] = ratios
        fails = []
        for build in sorted({k.split(".")[0] for k in ratios}):
            rs = {k: r for k, r in ratios.items() if k.startswith(build + ".")}
            geo = math.exp(sum(math.log(r) for r in rs.values()) / len(rs))
            print(
                f"##teamcity[buildStatisticValue key='sumd.{build}.geomean_ratio'"
                f" value='{geo:.4f}']"
            )
            print(f"{build}: geomean {geo:.4f} (min {a.geomean_min})")
            if geo < a.geomean_min:
                fails.append(f"{build} geomean {geo:.4f} < {a.geomean_min}")
            for k, r in sorted(rs.items()):
                flag = "REGRESSION" if r < a.cell_min else "ok"
                print(
                    f"  {k}: {cells[k]['median']:.1f} tok/s vs"
                    f" {base['cells'][k]['median']:.1f} = {r:.4f} {flag}"
                )
                if r < a.cell_min:
                    fails.append(f"{k} {r:.4f} < {a.cell_min}")
        for k, r in sorted(checks.items()):
            if base.get("nexttoken", {}).get(k) == "SAME" and r != "SAME":
                fails.append(f"next token {k}: SAME in the baseline, {r} now")
        result["fails"] = fails
        for f in fails:
            print(f"##teamcity[buildProblem description='{tc_escape(f)}']")
        if fails and status == 0:
            status = 1

    Path(a.out).write_text(json.dumps(result, indent=1) + "\n")
    print({0: "PASS", 1: "REGRESSION", 2: "BAD INPUT"}[status])
    return status


if __name__ == "__main__":
    sys.exit(main())
