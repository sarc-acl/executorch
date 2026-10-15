#!/usr/bin/env python3
"""Recompute fixed-count medians and uncertainty from preserved session rows."""
import csv
import json
import math
from pathlib import Path
import random
import statistics
import sys

root = Path(sys.argv[1])
REPS = 7  # owner decision 2026-10-15 23:30 UTC: 7 valid runs per arm
rows = list(csv.DictReader((root / "runs.csv").open()))
assert rows
required = {"device_uuid", "driver", "commit", "binary_sha256", "profile", "candidate", "session", "sensor_log", "valid", "reason"}
assert required <= rows[0].keys()
cells = []
ratios = []
rng = random.Random(58020261014)
for model in ("1b", "3b", "8b"):
    for scheme in ("4w", "8da4w"):
        cell = {"model": model, "scheme": scheme}
        arms = {}
        for arm in ("parent", "cand"):
            selected = [r for r in rows if r["model"] == model and r["scheme"] == scheme
                        and r["build"] == arm and r["log"].startswith("logs/prefill")]
            valid = [r for r in selected if r["valid"] == "1"]
            assert len(valid) >= REPS, (model, scheme, arm, "incomplete")
            used = valid[:REPS]
            values = []
            for row in used:
                log = (root / row["log"]).read_text(errors="replace")
                observations = [json.loads(line[line.index("{"):]) for line in log.splitlines() if "PyTorchObserver" in line]
                assert observations
                observation = observations[-1]
                assert observation["prompt_tokens"] == 2048 and observation["generated_tokens"] == 0
                rate = float(observation["prefill_token_per_sec"])
                assert rate == float(row["tok_s"]) and row["rc"] == "0" and row["reason"] == ""
                assert int(row["clk_n"]) >= 5 and float(row["fbusy_pct"]) == 0
                assert (root / row["sensor_log"]).is_file()
                values.append(rate)
            arms[arm] = values
            median = statistics.median(values)
            cell[arm] = {"median_tok_s": median, "used": REPS, "valid_saved": len(valid),
                         "invalid_saved": len(selected) - len(valid), "first_valid_repeats": [r["rep"] for r in used],
                         "spread_pct": 100 * (max(values) / min(values) - 1),
                         "median_clock_mhz": statistics.median(float(r["clk_med_mhz"]) for r in used)}
        ratio = cell["cand"]["median_tok_s"] / cell["parent"]["median_tok_s"]
        bootstrap = sorted(statistics.median(rng.choices(arms["cand"], k=REPS)) /
                           statistics.median(rng.choices(arms["parent"], k=REPS)) for _ in range(10000))
        cell["throughput_ratio"] = ratio
        cell["slowdown_pct"] = 100 * (1 - ratio)
        cell["unpaired_bootstrap_95pct"] = [bootstrap[249], bootstrap[9749]]
        cell["within_cell_bound"] = ratio >= 0.92
        cells.append(cell)
        ratios.append(ratio)
geomean = math.exp(statistics.mean(math.log(ratio) for ratio in ratios))
print(json.dumps({"cells": cells, "geomean_throughput_ratio": geomean,
                  "geomean_slowdown_pct": 100 * (1 - geomean),
                  "within_member_bounds": geomean >= 0.97 and all(cell["within_cell_bound"] for cell in cells),
                  "bootstrap_seed": 58020261014, "bootstrap_samples": 10000,
                  "acceptance": "bounds only; gate, raw sensor validity and dispatch review required"}, indent=2))
