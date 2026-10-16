"""Audit saved A/A rows, raw logs and observations without launching GPU work."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics


root = Path(os.environ["WMMA_CAMPAIGN_ROOT"])
artifacts = root / ".artifacts"
stage = artifacts / "stage/s1-aa-mm-r1"
change = root / "executorch/openspec/changes/sarc-1.5-wmma-families/nvidia-4070ti"
raw = (stage / "runs.csv").read_bytes()
assert raw.endswith(b"\n")
rows = list(csv.DictReader(raw.decode().splitlines(), strict=True))
reported = json.loads((stage / "calibration.json").read_text())
assert len(rows) == 84 and len({(r["cell"], r["arm"], r["rep"], r["attempt"]) for r in rows}) == 84
cells = [m + "-" + q for m in ["1b", "3b", "8b"] for q in ["4w", "8da4w"]]
minimum_samples, minimum_clock, maximum_clock = math.inf, math.inf, 0
record_hashes = []
for row in rows:
    log = (stage / row["log"]).read_text()
    stats = [json.loads(line[line.index("{"):]) for line in log.splitlines() if "PyTorchObserver" in line]
    assert len(stats) == 1
    stats = stats[0]
    record = json.loads((stage / row["record"]).read_text())
    samples = [json.loads(line) for line in (stage / row["samples"]).read_text().splitlines()]
    start, end = stats["inference_start_ms"], stats["prompt_eval_end_ms"]
    window = [s for s in samples if start * 1000 <= s["epoch_us"] <= end * 1000]
    clock = statistics.median(s["clock_mhz"] for s in window)
    rate = stats["prefill_token_per_sec"]
    thermal = sum(bool(s["event_reasons"] & 0x60) for s in samples)
    foreign = sum(any(c["own"] is not True for c in s["clients"]) for s in samples)
    assert int(row["rc"]) == record["returncode"] == 0
    assert stats["prompt_tokens"] == int(row["prompt_tokens"]) == 2048
    assert stats["generated_tokens"] == int(row["generated_tokens"]) == 0
    assert rate > 0 and math.isfinite(rate) and float(row["tok_s"]) == rate
    assert end > start > 0 and float(row["prefill_ms"]) == end - start
    assert len(window) == int(row["clock_samples"]) >= 5
    assert clock == float(row["clock_median_mhz"]) > 0
    assert thermal == int(row["thermal_samples"]) == 0
    assert foreign == int(row["foreign_samples"]) == 0
    assert len(samples) == int(row["sample_count"]) >= 2
    assert row["valid"] == "1" and not row["reason"] and not row["clock_floor_mhz"]
    assert not record["reasons"] and not record["slow_load_possible"] and record["resident_after"] >= record["model_bytes"]
    assert record["environment"] == {"ET_VK_SARC_UNVERIFIED": "1", "ET_VK_SARC_DEV_PROFILE": "4070ti-fused1", "ETVK_DEVICE_INDEX": "0"}
    assert record["commit"] == row["commit"] == (stage / row["arm"] / "COMMIT").read_text().strip()
    assert row["binary_sha256"] == record["binary_sha256"] == hashlib.sha256((stage / row["arm"] / "llama_main").read_bytes()).hexdigest()
    generated = "\n".join(line for line in log.splitlines() if "PyTorchObserver" not in line and not re.match(r"^[IWE] |^\[sarc_dev\]", line))
    assert hashlib.sha256(generated.encode()).hexdigest() == row["text_sha256"]
    minimum_samples = min(minimum_samples, len(window))
    minimum_clock, maximum_clock = min(minimum_clock, clock), max(maximum_clock, clock)
    record_hashes.append({"cell": row["cell"], "arm": row["arm"], "rep": int(row["rep"]),
                          "record_sha256": hashlib.sha256((stage / row["record"]).read_bytes()).hexdigest()})
for cell in cells:
    for rep in range(1, 8):
        pair = [r for r in rows if r["cell"] == cell and int(r["rep"]) == rep]
        assert [r["arm"] for r in pair] == (["parent", "topic"] if rep % 2 else ["topic", "parent"])
        assert pair[0]["text_sha256"] == pair[1]["text_sha256"]
calculated = {}
initial = {}
random_source = random.Random(407014)
historical = {}
with (root / "executorch/openspec/changes/sarc-1.5-4070ti-fused-port/results/4070ti/sessions/s5-pristine/summary.csv").open() as file:
    for row in csv.DictReader(file):
        if row["model"] in ["1b", "3b", "8b"]:
            historical[row["model"] + "-" + row["scheme"]] = float(row["cand_med_tok_s"])
for cell in cells:
    arms = {arm: [float(r["tok_s"]) for r in sorted(rows, key=lambda r: int(r["rep"])) if r["cell"] == cell and r["arm"] == arm] for arm in ["parent", "topic"]}
    assert all(len(rates) == 7 for rates in arms.values())
    for count, result in [(5, initial), (7, calculated)]:
        result[cell] = {"arms": {arm: {"median_tok_s": statistics.median(rates[:count]),
                                           "spread_fraction": (max(rates[:count]) - min(rates[:count])) / statistics.median(rates[:count])}
                                  for arm, rates in arms.items()}}
        result[cell]["ratio"] = result[cell]["arms"]["topic"]["median_tok_s"] / result[cell]["arms"]["parent"]["median_tok_s"]
    assert initial[cell] == reported["initial_five"][cell]
    result = calculated[cell]
    ratios = []
    for _ in range(20000):
        indices = [random_source.randrange(7) for _ in range(7)]
        ratios.append(statistics.median(arms["topic"][i] for i in indices) / statistics.median(arms["parent"][i] for i in indices))
    ratios.sort()
    interval = []
    for probability in [0.025, 0.975]:
        position = (len(ratios) - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        interval.append(ratios[lower] + (ratios[upper] - ratios[lower]) * (position - lower))
    result.update(ratio_interval95=interval, historical_tok_s=historical[cell], historical_ratio=result["arms"]["parent"]["median_tok_s"] / historical[cell])
    assert result == reported["cells"][cell], cell
assert any(abs(c["ratio"] - 1) > 0.01 or any(a["spread_fraction"] > 0.03 for a in c["arms"].values()) for c in initial.values())
floor = math.floor(0.97 * minimum_clock)
geomean = math.prod(c["ratio"] for c in calculated.values()) ** (1 / 6)
assert floor == reported["clock_floor_mhz"] and reported["repeats"] == 7
assert math.isclose(geomean, reported["geomean_ratio"], abs_tol=1e-15)
assert abs(geomean - 1) <= 0.01 and all(abs(c["ratio"] - 1) <= 0.02 for c in calculated.values())
assert all(abs(c["historical_ratio"] - 1) <= 0.03 for c in calculated.values())
cleanup = [json.loads(line) for line in (stage / "child-cleanup.jsonl").read_text().splitlines()]
assert len(cleanup) == 3 and all(not r["remaining_descendants"] for r in cleanup)
summary = {"session": stage.name, "source": "actor recomputation from every saved log/record/sample; independent review pending",
           "valid_rows": 84, "invalid_rows": 0, "repeats": 7, "clock_floor_mhz": floor,
           "minimum_run_clock_mhz": minimum_clock, "maximum_run_clock_mhz": maximum_clock,
           "minimum_prefill_samples": minimum_samples, "geomean_ratio": geomean,
           "aa_within_limits": True, "historical_outside_3pct": [], "cells": calculated,
           "csv_sha256": hashlib.sha256(raw).hexdigest(),
           "calibration_sha256": hashlib.sha256((stage / "calibration.json").read_bytes()).hexdigest(),
           "cleanup_checks": cleanup, "candidate_matrix": "not started"}
(artifacts / "checks/calibration-audit.json").write_text(json.dumps(dict(summary, record_hashes=record_hashes), indent=2) + "\n")
(change / "results/calibration.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: summary[k] for k in ["valid_rows", "clock_floor_mhz", "minimum_prefill_samples", "geomean_ratio", "aa_within_limits", "historical_outside_3pct"]}))
