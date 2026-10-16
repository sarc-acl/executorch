"""Capture the finite eleven-profile gate list, preserving failures and coverage."""

import csv
import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


root = Path(os.environ["WMMA_CAMPAIGN_ROOT"])
artifacts = root / ".artifacts"
change = root / "executorch/openspec/changes/sarc-1.5-wmma-families/nvidia-4070ti"
tools = Path(__file__).resolve().parent
manifest_path = root / "inputs/inventory-f393c3ff7f/manifests/nvidia-ampere-ada.json"
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == "ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1"
manifest = json.loads(manifest_path.read_text())
input_csv = subprocess.check_output(
    ["git", "-C", str(root / "executorch"), "show",
     os.environ["WMMA_ORCHESTRATION_COMMIT"] + ":openspec/changes/sarc-1.5-wmma-families/nvidia-4070ti/results/candidate-preflight.csv"],
    text=True)
candidates = list(csv.DictReader(input_csv.splitlines()))
assert len(candidates) == 11
assert {c["shader"] for c in candidates} == {c["shader"] for c in manifest["candidates"]}
assert [c["candidate"] for c in candidates] == [f"4070ti-wmma-c{i:02d}" for i in range(1, 12)]
calibration_path = artifacts / "stage/s1-aa-mm-r1/calibration.json"
assert hashlib.sha256(calibration_path.read_bytes()).hexdigest() == json.loads((change / "results/calibration.json").read_text())["calibration_sha256"]
calibration = json.loads(calibration_path.read_text())
assert calibration["aa_within_limits"] and calibration["repeats"] == 7 and calibration["clock_floor_mhz"] == 2531
assert not calibration["historical_outside_3pct"]
reference = (artifacts / "stage/s0-topic-mm-control/verify.out").read_text()
results = artifacts / "matrix/gates-r1"
results.mkdir(parents=True, exist_ok=True)
summary_path = results / "summary.json"
summaries = json.loads(summary_path.read_text()) if summary_path.exists() else []


def normalized(text):
    return re.sub(r"(prefill_tok_s|decode_tok_s)=[0-9.]*", r"\1=<RATE>", text)


for candidate in candidates:
    profile = candidate["candidate"]
    if any(s["profile"] == profile for s in summaries):
        continue
    stage = artifacts / "stage" / ("s2-gate-" + profile.rsplit("-", 1)[-1])
    assert not stage.exists(), "partial/unassessed gate exists; diagnose before any retry"
    assert not (artifacts / "GPU_GONE").exists()
    env = dict(os.environ, WMMA_MATRIX_PROFILE=profile, WMMA_CANDIDATE=profile,
               WMMA_PURPOSE="candidate_gate", WMMA_SESSION=stage.name)
    subprocess.run(["python3", str(tools / "parent_snapshot.py"), "matrix-mm-r1", stage.name], env=env, check=True)
    records = [json.loads(p.read_text()) for p in sorted((stage / "observations").glob("*/record.json"))]
    assert len(records) == 37 and all(not r["reasons"] for r in records)
    assert all(r["environment"].get("ET_VK_SARC_WMMA_PROFILE") == profile for r in records)
    sensor_samples, thermal, foreign = 0, 0, 0
    for path in (stage / "observations").glob("*/samples.jsonl"):
        for line in path.open():
            sample = json.loads(line)
            sensor_samples += 1
            thermal += bool(sample["event_reasons"] & 0x60)
            foreign += any(c["own"] is not True for c in sample["clients"])
    assert sensor_samples > 0 and thermal == foreign == 0
    microbench = [r for r in records if "model" not in r]
    assert len(microbench) == 15 and all(r["returncode"] in [0, 1] for r in microbench), "unexpected microbench exit; diagnose"
    correctness = next(r for r in microbench if "--correctness-only" in r["argv"])
    pdiff_records = [r for r in microbench if "--production-diff" in r["argv"]]
    assert len(pdiff_records) == 12
    pdiff_cases, numeric_failures, details = [], [], []
    for path in sorted((stage / "verify").glob("pdiff-*.log")):
        text = path.read_text()
        for match in re.finditer(r"^\[production-diff\] (.*?) -> (\S+) .*?, correctness=(.*)$", text, re.MULTILINE):
            pdiff_cases.append({"log": str(path.relative_to(stage)), "case": match[1], "shader": match[2], "correctness": match[3]})
        details.extend({"log": str(path.relative_to(stage)), "line": line} for line in text.splitlines()
                       if "[sampled reference]" in line or "mismatch" in line.lower() or "FAILED" in line)
        if "[production-diff] ALL PASSED" not in text:
            numeric_failures.append(str(path.relative_to(stage)))
    assert len(list((stage / "verify").glob("pdiff-*.log"))) == 12
    if correctness["returncode"] != 0:
        numeric_failures.append("verify/correctness.log")
    command_failures = [{"argv": r["argv"], "returncode": r["returncode"]}
                        for r in [correctness, *pdiff_records] if r["returncode"] != 0]
    linear = []
    for scheme in ["4w", "8da4w"]:
        cases = json.loads((stage / "verify" / ("linear-" + scheme + ".json")).read_text())["cases"]
        assert len(cases) == 24
        linear.extend(cases)
    linear_not_ok = [row for row in linear if not row["ok"]]
    output = (stage / "verify.out").read_text()
    diff = "".join(difflib.unified_diff(normalized(reference).splitlines(True), normalized(output).splitlines(True), fromfile="control", tofile=profile))
    (stage / "control-output.diff").write_text(diff)
    clean_reference = [line.split(" kernels:")[0] if line.startswith("linear ") else line for line in normalized(reference).splitlines()]
    clean_current = [line.split(" kernels:")[0] if line.startswith("linear ") else line for line in normalized(output).splitlines()]
    non_dispatch_differences = [{"control": before, "candidate": after} for before, after in zip(clean_reference, clean_current) if before != after]
    if len(clean_reference) != len(clean_current):
        non_dispatch_differences.append({"control_lines": len(clean_reference), "candidate_lines": len(clean_current)})
    expected = candidate["shader"]
    coverage = {"linear_benchmark_cases": sum(row["kernel"] == expected for row in linear),
                "production_diff_cases": sum(row["shader"] == expected for row in pdiff_cases)}
    verdict = "CAPTURE_COMPLETE; NEXT_TOKEN_AND_SESSION_EVIDENCE_PENDING"
    if not coverage["production_diff_cases"]:
        verdict = "NO_PRODUCTION_DIFF_CANDIDATE_DISPATCH; ALTERNATE_COVERAGE_PENDING"
    if non_dispatch_differences:
        verdict = "CONTROL_DIFFERENCES_REQUIRE_DIAGNOSIS; NO_ACCEPTANCE"
    if linear_not_ok:
        verdict = "LINEAR_BENCHMARK_GATE_FAILED; DIAGNOSIS_PENDING"
    if numeric_failures or command_failures:
        verdict = "CORRECTNESS_GATE_FAILED; NO_RETRY_OR_ACCEPTANCE"
    row = {"profile": profile, "shader": expected, "scheme": candidate["scheme"],
           "io_storage": candidate["io_storage"], "weight_storage": candidate["weight_storage"],
           "stage": str(stage.relative_to(artifacts)), "commit": (stage / "COMMIT").read_text().strip(),
           "verdict": verdict, "coverage": coverage, "numeric_failures": numeric_failures,
           "correctness_returncode": correctness["returncode"],
           "command_failures": command_failures,
           "production_diff_returncodes": [r["returncode"] for r in pdiff_records],
           "linear_returncodes": [r["returncode"] for r in microbench if "--linear" in r["argv"]],
           "linear_not_ok": linear_not_ok,
           "non_dispatch_differences": non_dispatch_differences,
           "thermal_samples": thermal, "foreign_samples": foreign,
           "production_diff_cases": pdiff_cases, "linear_cases": linear,
           "numeric_details": details, "default_tiled_differences": output.count("output DIFFER"),
           "timing": "NOT_RUN", "acceptance": "NONE",
           "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (stage / "candidate-assessment.json").write_text(json.dumps(row, indent=2) + "\n")
    summaries.append({k: row[k] for k in ["profile", "shader", "scheme", "io_storage", "weight_storage", "stage", "commit", "verdict", "coverage", "numeric_failures", "non_dispatch_differences", "default_tiled_differences", "timing", "acceptance", "updated_utc"]})
    temporary = summary_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summaries, indent=2) + "\n")
    temporary.replace(summary_path)
    (change / "results/candidate-gates.json").write_text(json.dumps(summaries, indent=2) + "\n")
    status = change / "STATUS.md"
    with status.open("a") as file:
        file.write("\n" + row["updated_utc"] + " Candidate " + profile + ": " + verdict + ". Coverage " + json.dumps(coverage) +
                   "; assessment artifact " + str((stage / "candidate-assessment.json").relative_to(artifacts)) + ". No acceptance/timing.\n")
    print(profile, verdict, coverage, flush=True)
print("FINITE_GATE_CAPTURES_COMPLETE; inspect assessments before any timing or retry", flush=True)
