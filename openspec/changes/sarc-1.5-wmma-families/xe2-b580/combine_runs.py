#!/usr/bin/env python3
"""combine_runs.py <artifacts> <change dir>: combine every session's runs.csv and the per-candidate analyses.

Writes <artifacts>/results/runs.csv (all columns), <change>/results/runs.csv (without host and device identity),
<change>/results/matrix-summary.csv and <change>/results/cell-coverage.csv. `used` marks the rows that
analyze_session.py takes (first 7 valid rows at or below BUSYMAX per arm and cell of the analysed session)."""
import csv
import json
from pathlib import Path
import sys

A, C = Path(sys.argv[1]), Path(sys.argv[2])
busymax = float((A / "busymax_pct").read_text())
CAVEAT = ("Intel caveat: texture3d images above ANV's 2048 extent (RoPE tables in all six models, 8B activations) "
          "are pre-existing and identical in both arms (owner decision 2026-10-15 23:30 UTC)")
SESSIONS = [
    ("s1-aa", "raw", "superseded: A/A under the zero foreign ceiling, all prefill rows invalid"),
    ("s1c-aa", "raw", "A/A calibration session (no busy limit at run time; BUSYMAX applied in analysis)"),
    ("s2-b70-linear", "raw", ""), ("s2b-b70-linear", "raw-8b-4w", "re-time of s2-b70-linear 8b-4w"),
    ("s3-dq-n26", "raw", ""), ("s4-q4-narrow", "raw", ""), ("s5-q4-band", "raw", ""),
    ("s6-dq-table", "raw", ""), ("s7-dq-i", "raw", ""),
]
ANALYSES = {"aa": "s1c-aa", "b70-linear": "s2-b70-linear", "dq-n26": "s3-dq-n26", "q4-narrow": "s4-q4-narrow",
            "q4-band": "s5-q4-band", "dq-table": "s6-dq-table", "dq-i": "s7-dq-i"}
analyses = {c: json.loads((A / "stage" / s / "analysis.json").read_text()) for c, s in ANALYSES.items()}
sources = {(Path(cell["source"]).parent.name, Path(cell["source"]).name, cell["model"], cell["scheme"])
           for a in analyses.values() for cell in a["cells"]}

rows, fields = [], []
for session, out, note in SESSIONS:
    table = list(csv.DictReader((A / "stage" / session / out / "runs.csv").open()))
    counts = {}
    for r in table:
        r = {"session_dir": f"{session}/{out}", **r}
        timed = r["log"].startswith("logs/prefill")
        ok = timed and r["valid"] == "1" and r["fbusy_pct"] != "" and float(r["fbusy_pct"]) <= busymax
        key = (r["model"], r["scheme"], r["build"])
        used = ok and (session, out, r["model"], r["scheme"]) in sources and counts.get(key, 0) < 7
        if used:
            counts[key] = counts.get(key, 0) + 1
        r["used"] = int(used)
        r["notes"] = "; ".join(x for x in (note, f"BUSYMAX {busymax} %", CAVEAT) if x)
        rows.append(r)
        if not fields:
            fields = list(r)
for r in rows:
    for f in r:
        if f not in fields:
            fields.insert(fields.index("used"), f)
(A / "results").mkdir(exist_ok=True)
with (A / "results" / "runs.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fields, restval=""); w.writeheader(); w.writerows(rows)
public = [f for f in fields if f not in ("host", "device_uuid")]
with (C / "results" / "runs.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, public, restval="", extrasaction="ignore"); w.writeheader(); w.writerows(rows)

with (C / "results" / "matrix-summary.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["candidate", "session", "model", "scheme", "control_median_tok_s", "candidate_median_tok_s",
                "ratio", "bootstrap95_lo", "bootstrap95_hi", "control_spread_pct", "candidate_spread_pct",
                "valid_control", "valid_candidate", "invalid_control", "invalid_candidate", "cell_ok", "geomean",
                "member_bounds_ok"])
    for cand, a in analyses.items():
        for c in a["cells"]:
            w.writerow([cand, Path(c["source"]).parent.name, c["model"], c["scheme"],
                        c["parent"]["median_tok_s"], c["cand"]["median_tok_s"], round(c["throughput_ratio"], 4),
                        round(c["unpaired_bootstrap_95pct"][0], 4), round(c["unpaired_bootstrap_95pct"][1], 4),
                        round(c["parent"]["spread_pct"], 2), round(c["cand"]["spread_pct"], 2),
                        c["parent"]["valid_saved"], c["cand"]["valid_saved"], c["parent"]["invalid_saved"],
                        c["cand"]["invalid_saved"], c["within_cell_bound"], round(a["geomean_throughput_ratio"], 4),
                        a["within_member_bounds"]])

with (C / "results" / "cell-coverage.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "scheme", "control", "candidate", "status", "timed_runs", "valid_used"])
    for cand, a in analyses.items():
        for c in a["cells"]:
            src = (Path(c["source"]).parent.name, Path(c["source"]).name)
            n = sum(1 for r in rows if r["session_dir"] == "/".join(src) and r["model"] == c["model"]
                    and r["scheme"] == c["scheme"] and r["log"].startswith("logs/prefill"))
            w.writerow([c["model"], c["scheme"], "b580-fused1-order-p13", cand, "MEASURED", n,
                        sum(int(r["used"]) for r in rows if r["session_dir"] == "/".join(src)
                            and r["model"] == c["model"] and r["scheme"] == c["scheme"])])
print(len(rows), "rows;", sum(r["used"] for r in rows), "used")
