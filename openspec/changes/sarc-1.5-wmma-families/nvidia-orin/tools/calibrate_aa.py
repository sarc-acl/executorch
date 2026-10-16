#!/usr/bin/env python3
"""Calibrate the registered per-run clock rule from fully valid A/A rows."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def validity(row, floor=None):
    reasons = []
    provenance = ("device", "unit", "driver", "commit", "binary_sha256", "profile",
                  "candidate", "session", "cell", "arm", "sensor_path", "log")
    reasons.extend("missing_" + name for name in provenance if not row.get(name))
    try:
        if int(row["rc"]) != 0:
            reasons.append("rc")
        rate = float(row["tok_s"])
        if not math.isfinite(rate) or rate <= 0:
            reasons.append("rate")
        if int(row["prompt_tokens"]) != 2048 or int(row["generated_tokens"]) != 0:
            reasons.append("tokens")
        if int(row["prefill_clock_samples"]) < 5:
            reasons.append("clock_samples")
        clock = float(row["prefill_clock_median_mhz"])
        if not math.isfinite(clock) or clock <= 0 or (floor is not None and clock < floor):
            reasons.append("clock")
    except (KeyError, ValueError, TypeError):
        reasons.append("missing_or_invalid_stats")
    if any(row.get(name) != "0" for name in ("foreign_before", "foreign_during", "foreign_after")):
        reasons.append("foreign_workload")
    if row.get("thermal_throttle") != "0":
        reasons.append("thermal_throttle")
    if row.get("sensor_error") != "0":
        reasons.append("sensor_error")
    return reasons


def calibrate(rows):
    preliminary = [row for row in rows if not validity(row)]
    if not preliminary:
        raise ValueError("no fully valid A/A rows")
    floor = 0.97 * min(float(row["prefill_clock_median_mhz"]) for row in preliminary)
    decisions = [{"row_index": i, "reasons": validity(row, floor)} for i, row in enumerate(rows)]
    expected = {(arm, model + "-" + scheme) for arm in ("control", "candidate")
                for model in ("1b", "3b", "8b") for scheme in ("4w", "8da4w")}
    groups = {}
    for row, decision in zip(rows, decisions):
        if not decision["reasons"]:
            groups.setdefault((row["arm"], row["cell"]), []).append(float(row["tok_s"]))
    if set(groups) != expected or any(len(values) != 5 for values in groups.values()):
        raise ValueError("A/A requires exactly five fully valid rows in each of twelve arm/cell groups")
    return {"clock_floor_mhz": floor, "clock_rule": "0.97 * lowest fully valid per-run median",
            "fixed_noise_band_percent": 2, "valid_rows_per_arm_cell": 5,
            "medians": {arm + ":" + cell: statistics.median(values)
                        for (arm, cell), values in groups.items()}, "decisions": decisions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    with args.runs_csv.open(newline="") as file:
        result = calibrate(list(csv.DictReader(file)))
    with args.output_json.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
