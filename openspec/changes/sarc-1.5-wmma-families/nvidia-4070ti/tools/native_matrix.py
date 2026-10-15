"""Run exactly the seven native manifest profiles after inspecting all gate captures."""

import datetime
import json
import os
from pathlib import Path
import subprocess


root = Path(os.environ["WMMA_CAMPAIGN_ROOT"])
artifacts = root / ".artifacts"
change = root / "executorch/openspec/changes/sarc-1.5-wmma-families/nvidia-4070ti"
tools = Path(__file__).resolve().parent
profiles = ["4070ti-wmma-c08", "4070ti-wmma-c09", "4070ti-wmma-c10", "4070ti-wmma-c11", "4070ti-wmma-c06", "4070ti-wmma-c01", "4070ti-wmma-c02"]
results = artifacts / "matrix/native-r1"
results.mkdir(parents=True, exist_ok=True)
summary_path = results / "summary.json"
summaries = json.loads(summary_path.read_text()) if summary_path.exists() else []
for profile in profiles:
    if any(r["profile"] == profile for r in summaries):
        continue
    assert not (artifacts / "GPU_GONE").exists()
    stage = artifacts / "stage" / ("s3-" + profile)
    if not (stage / "summary.json").exists():
        subprocess.run(["timeout", "--signal=TERM", "--kill-after=60", "4h", "python3", str(tools / "candidate_session.py"), profile], check=True)
    summary = json.loads((stage / "summary.json").read_text())
    assert summary["profile"] == profile
    summaries.append(dict(summary, stage=str(stage.relative_to(artifacts)), updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    temporary = summary_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summaries, indent=2) + "\n")
    temporary.replace(summary_path)
    (change / "results/native-sessions.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (change / "STATUS.md").open("a") as file:
        file.write("\n" + summaries[-1]["updated_utc"] + " " + profile + ": " + summary["status"] + "; correctness=" + summary.get("correctness", "n/a") + "; acceptance=" + summary["acceptance"] + ".")
        if "cells" in summary:
            file.write(" Ratios " + json.dumps({cell: row["ratio"] for cell, row in summary["cells"].items()}) + "; geomean=" + str(summary["geomean_ratio"]) + ".")
        file.write(" Evidence artifact " + str(stage.relative_to(artifacts)) + ".\n")
    print(profile, summary["status"], summary.get("geomean_ratio"), flush=True)
print("FINITE_NATIVE_QUEUE_COMPLETE; raw audit, alternate coverage and independent review remain", flush=True)
