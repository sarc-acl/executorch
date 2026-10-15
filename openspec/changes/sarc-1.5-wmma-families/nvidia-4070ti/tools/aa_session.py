"""Resumable six-cell A/A using the fixed calibration and measurement rules."""

import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from evaluate_run import evaluate
from nvml_observer import Observer
from session_control import hold_point


root = Path(os.environ["WMMA_CAMPAIGN_ROOT"])
artifacts = root / ".artifacts"
tools = Path(__file__).resolve().parent
stage = artifacts / "stage/s1-aa"
stage.mkdir(parents=True, exist_ok=True)
session_lock = (stage / "session.lock").open("a")
fcntl.flock(session_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
models = json.loads((artifacts / "preflight/models-static.json").read_text())
builds = {"parent": "parent-r1", "topic": "matrix-r1"}
env = {k: v for k, v in os.environ.items() if not k.startswith(("ET_VK", "ETVK"))}
env.update(ET_VK_SARC_UNVERIFIED="1", ET_VK_SARC_DEV_PROFILE="4070ti-fused1",
           ETVK_DEVICE_INDEX=os.environ["WMMA_DEVICE_INDEX"], WMMA_PURPOSE="calibration",
           WMMA_SESSION="s1-aa", WMMA_CANDIDATE="control", PYTHONDONTWRITEBYTECODE="1",
           SARC_CAMPAIGN_TAG="wmma-nvidia-4070ti-20261015-aa", TMPDIR=str(artifacts / "tmp"))
assert not (artifacts / "GPU_GONE").exists()
assert not (artifacts / "stage/s0-topic-control/ABORTED").exists()
control_gate = json.loads((artifacts / "stage/s0-topic-control/snapshot-summary.json").read_text())
assert control_gate["observations"] == 37 and not control_gate["aborted"]
envs, binaries = {}, {}
for arm, tag in builds.items():
    build = artifacts / "build" / tag
    arm_stage = stage / arm
    arm_stage.mkdir(exist_ok=True)
    for source, name in [(build / "llama/examples/models/llama/llama_main", "llama_main"),
                         (build / "llama/examples/models/llama/runner/libllama_runner.so", "libllama_runner.so"),
                         (build / "COMMIT", "COMMIT")]:
        target = arm_stage / name
        if target.exists():
            assert target.read_bytes() == source.read_bytes(), target
        else:
            shutil.copy2(source, target)
    arm_tools = arm_stage / "observer-tools"
    arm_tools.mkdir(exist_ok=True)
    for name in ["observe.py", "nvml_observer.py", "warm_file.py"]:
        target = arm_tools / name
        if target.exists():
            assert target.read_bytes() == (tools / name).read_bytes(), target
        else:
            shutil.copy2(tools / name, target)
    envs[arm] = dict(env, WMMA_STAGE=str(arm_stage), LD_LIBRARY_PATH=str(arm_stage))
    binaries[arm] = {"sha256": hashlib.sha256((arm_stage / "llama_main").read_bytes()).hexdigest(),
                     "commit": (arm_stage / "COMMIT").read_text().strip()}

observer = Observer(int(os.environ["WMMA_DEVICE_INDEX"]), os.getpid())
assert observer.identity["unit"] == "GPU-" + os.environ["WMMA_DEVICE_LOCK"]
device_lock = (Path.home() / ".cache/gpu-lab" / ("lock-" + os.environ["WMMA_DEVICE_LOCK"])).open("a")
rows_path = stage / "runs.csv"
fields = ["unit", "device_index", "driver", "session", "cell", "arm", "candidate", "profile", "commit", "binary_sha256",
          "rep", "attempt", "tok_s", "prompt_tokens", "generated_tokens", "prefill_ms", "clock_samples", "clock_median_mhz",
          "clock_floor_mhz", "sample_count", "thermal_samples", "foreign_samples", "valid", "reason", "text_sha256",
          "rc", "validity_mode", "log", "record", "samples", "utc"]
if rows_path.exists():
    raw = rows_path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        cutoff = raw.rfind(b"\n") + 1
        superseded = stage / "superseded/truncated-csv"
        superseded.mkdir(parents=True, exist_ok=True)
        (superseded / str(time.time_ns())).write_bytes(raw[cutoff:])
        rows_path.write_bytes(raw[:cutoff])
    with rows_path.open() as file:
        rows = list(csv.DictReader(file, strict=True))
else:
    rows = []
    with rows_path.open("w", newline="") as file:
        csv.DictWriter(file, fieldnames=fields, lineterminator="\n").writeheader()


def cool(target, maximum):
    temperatures = []
    deadline = time.monotonic() + maximum
    while True:
        sample = observer.sample()
        if any(c["own"] is not True for c in sample["clients"]):
            raise RuntimeError("foreign GPU client while cooling; no job launched")
        temperatures.append(sample["temperature_c"])
        if temperatures[-1] <= target or time.monotonic() >= deadline or (
                len(temperatures) >= 3 and temperatures[-1] >= temperatures[-3]):
            return temperatures[-1]
        time.sleep(5)


def valid_rows(cell, arm, count):
    return [r for r in rows if r["cell"] == cell and r["arm"] == arm and int(r["rep"]) <= count and int(r["valid"]) == 1]


def save_row(row):
    with rows_path.open("a", newline="") as file:
        csv.DictWriter(file, fieldnames=fields, lineterminator="\n").writerow(row)
        file.flush()
        os.fsync(file.fileno())
    rows.append(row)


def run_rounds(first, last, target):
    for model in models:
        path = Path(model["path"])
        cell = ("1b" if "-1b" in path.stem else "3b" if "-3b" in path.stem else "8b") + ("-8da4w" if "8da4w" in path.stem else "-4w")
        tokenizer = path.parent.parent / "original/tokenizer.model"
        for rep in range(first, last + 1):
            order = ["parent", "topic"] if rep % 2 else ["topic", "parent"]
            for attempt in range(1, 5):
                for arm in order:
                    if any(r["cell"] == cell and r["arm"] == arm and int(r["rep"]) == rep and int(r["valid"]) for r in rows):
                        continue
                    hold_point(device_lock, artifacts)
                    assert not (artifacts / "GPU_GONE").exists()
                    assert not (stage / arm / "ABORTED").exists()
                    key = f"{cell}-{arm}-r{rep}-a{attempt}"
                    if any(r["cell"] == cell and r["arm"] == arm and int(r["rep"]) == rep and int(r["attempt"]) == attempt for r in rows):
                        continue
                    run_dir = stage / "raw" / key
                    log = stage / "logs" / (key + ".log")
                    log.parent.mkdir(exist_ok=True)
                    if run_dir.exists() or log.exists():
                        superseded = stage / "superseded/interrupted" / (key + "-" + str(time.time_ns()))
                        superseded.mkdir(parents=True)
                        if run_dir.exists():
                            shutil.move(str(run_dir), superseded / "raw")
                        if log.exists():
                            shutil.move(str(log), superseded / "runner.log")
                        save_row(dict(observer.identity, session="s1-aa", cell=cell, arm=arm, candidate="control",
                                      profile="4070ti-fused1", commit=binaries[arm]["commit"], binary_sha256=binaries[arm]["sha256"],
                                      rep=rep, attempt=attempt, valid=0, reason="interrupted_before_csv_append",
                                      validity_mode="A/A_CALIBRATION", log=str((superseded / "runner.log").relative_to(stage)),
                                      record=str((superseded / "raw/record.json").relative_to(stage)),
                                      samples=str((superseded / "raw/samples.jsonl").relative_to(stage)),
                                      utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
                        continue
                    cool(target, 120)
                    child_env = dict(envs[arm], WMMA_RECORD_DIR=str(run_dir))
                    command = ["timeout", "--signal=TERM", "--kill-after=30", "1200", "python3",
                               str(stage / arm / "observer-tools/observe.py"), str(stage / arm / "llama_main"),
                               "--model_path", str(path), "--tokenizer_path", str(tokenizer),
                               "--prompt_file", str(artifacts / "stage/s0-parent-verify/prompt_2048.txt"),
                               "--max_new_tokens", "1", "--temperature", "0", "--warmup"]
                    with log.open("w") as output:
                        result = subprocess.run(command, env=child_env, stdin=subprocess.DEVNULL,
                                                stdout=output, stderr=subprocess.STDOUT)
                    descendants = set()
                    if len(rows) < 3:
                        processes = {}
                        for entry in Path("/proc").iterdir():
                            if not entry.name.isdigit():
                                continue
                            try:
                                stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                                processes[int(entry.name)] = int(stat[1])
                            except (FileNotFoundError, ProcessLookupError):
                                continue
                        descendants = {os.getpid()}
                        while True:
                            found = {pid for pid, parent in processes.items() if parent in descendants}
                            if found <= descendants:
                                break
                            descendants.update(found)
                        descendants.remove(os.getpid())
                        with (stage / "child-cleanup.jsonl").open("a") as output:
                            output.write(json.dumps({"run": key, "remaining_descendants": sorted(descendants)}) + "\n")
                    try:
                        metrics = evaluate(log, run_dir / "samples.jsonl", result.returncode)
                    except Exception as error:
                        metrics = {"valid": 0, "reason": "evaluation_failed:" + str(error)}
                    if descendants:
                        metrics.update(valid=0, reason="sampling_or_runner_child_remains")
                    row = dict(observer.identity, session="s1-aa", cell=cell, arm=arm, candidate="control",
                               profile="4070ti-fused1", commit=binaries[arm]["commit"], binary_sha256=binaries[arm]["sha256"],
                               rep=rep, attempt=attempt, rc=result.returncode, validity_mode="A/A_CALIBRATION",
                               log=str(log.relative_to(stage)), record=str((run_dir / "record.json").relative_to(stage)),
                               samples=str((run_dir / "samples.jsonl").relative_to(stage)),
                               utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **metrics)
                    save_row(row)
                    print(cell, arm, rep, attempt, metrics, flush=True)
                    assert not descendants, "sampling/runner child remains after the run"
                    if result.returncode != 0 or (stage / arm / "ABORTED").exists():
                        raise RuntimeError("observed command failed; diagnose the retained row before retry")
                if all(any(r["cell"] == cell and r["arm"] == arm and int(r["rep"]) == rep and int(r["valid"]) for r in rows) for arm in order):
                    pair = [next(r for r in rows if r["cell"] == cell and r["arm"] == arm and int(r["rep"]) == rep and int(r["valid"])) for arm in order]
                    assert pair[0]["text_sha256"] == pair[1]["text_sha256"], "A/A output difference"
                    break
            else:
                raise RuntimeError(f"four attempts without valid A/A pair: {cell} r{rep}")


cells = [model + "-" + scheme for model in ["1b", "3b", "8b"] for scheme in ["4w", "8da4w"]]


def medians(count):
    result = {}
    for cell in cells:
        arms = {}
        for arm in builds:
            rates = [float(r["tok_s"]) for r in valid_rows(cell, arm, count)]
            assert len(rates) == count
            median = statistics.median(rates)
            arms[arm] = {"median_tok_s": median, "spread_fraction": (max(rates) - min(rates)) / median}
        result[cell] = {"arms": arms, "ratio": arms["topic"]["median_tok_s"] / arms["parent"]["median_tok_s"]}
    return result


try:
    hold_point(device_lock, artifacts)
    if (stage / "STAGE.json").exists():
        manifest = json.loads((stage / "STAGE.json").read_text())
        assert manifest["binaries"] == binaries
        target = manifest["cool_target_c"]
    else:
        idle = cool(50, 300)
        target = idle + 5
        manifest = {"purpose": "control A/A and historical tuned reference", "binaries": binaries,
                    "device": observer.identity, "cool_target_c": target,
                    "cool_rule": "initial <=50C/300s; then idle+5C/120s; stop after three 5s probes stop falling",
                    "protocol": {"initial_repeats": 5, "extended_repeats": 7, "minimum_samples": 5,
                                 "ratio_extend": 0.01, "spread_extend": 0.03,
                                 "clock_floor_rule": "floor(0.97*lowest_valid_run_median)", "replacement_attempts": 4},
                    "orchestration_commit": os.environ.get("WMMA_ORCHESTRATION_COMMIT") or subprocess.check_output(["git", "-C", str(root / "executorch"), "rev-parse", "HEAD"], text=True).strip()}
        (stage / "STAGE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run_rounds(1, 5, target)
    first = medians(5)
    extended = any(abs(c["ratio"] - 1) > 0.01 or any(a["spread_fraction"] > 0.03 for a in c["arms"].values()) for c in first.values())
    repeats = 7 if extended else 5
    if extended:
        run_rounds(6, 7, target)
    final = medians(repeats)
    valid = [r for r in rows if int(r["valid"]) and int(r["rep"]) <= repeats]
    floor = math.floor(0.97 * min(float(r["clock_median_mhz"]) for r in valid))
    random_source = random.Random(407014)
    historical = {}
    reference = root / "executorch/openspec/changes/sarc-1.5-4070ti-fused-port/results/4070ti/sessions/s5-pristine/summary.csv"
    with reference.open() as file:
        for row in csv.DictReader(file):
            if row["model"] in ["1b", "3b", "8b"]:
                historical[row["model"] + "-" + row["scheme"]] = float(row["cand_med_tok_s"])
    for cell, result in final.items():
        parent = sorted(valid_rows(cell, "parent", repeats), key=lambda r: int(r["rep"]))
        topic = sorted(valid_rows(cell, "topic", repeats), key=lambda r: int(r["rep"]))
        intervals = []
        for _ in range(20000):
            indices = [random_source.randrange(repeats) for _ in range(repeats)]
            intervals.append(statistics.median(float(topic[i]["tok_s"]) for i in indices) /
                             statistics.median(float(parent[i]["tok_s"]) for i in indices))
        intervals.sort()
        bounds = []
        for probability in [0.025, 0.975]:
            position = (len(intervals) - 1) * probability
            left, right = math.floor(position), math.ceil(position)
            bounds.append(intervals[left] + (intervals[right] - intervals[left]) * (position - left))
        result.update(ratio_interval95=bounds, historical_tok_s=historical[cell],
                      historical_ratio=result["arms"]["parent"]["median_tok_s"] / historical[cell])
    geomean = math.exp(statistics.mean(math.log(c["ratio"]) for c in final.values()))
    accepted = abs(geomean - 1) <= 0.01 and all(abs(c["ratio"] - 1) <= 0.02 for c in final.values())
    calibration = {"repeats": repeats, "clock_floor_mhz": floor, "geomean_ratio": geomean,
                   "aa_within_limits": accepted, "cells": final, "initial_five": first,
                   "invalid_rows": sum(not int(r["valid"]) for r in rows),
                   "historical_outside_3pct": [cell for cell, c in final.items() if abs(c["historical_ratio"] - 1) > 0.03],
                   "candidate_launch": "NOT_AUTHORIZED_BY_THIS_RESULT; inspect A/A and explain baseline differences first"}
    (stage / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    print(json.dumps(calibration), flush=True)
    assert accepted, "A/A outside fixed limits; diagnose, do not widen them"
finally:
    fcntl.flock(device_lock, fcntl.LOCK_UN)
    device_lock.close()
    observer.close()
