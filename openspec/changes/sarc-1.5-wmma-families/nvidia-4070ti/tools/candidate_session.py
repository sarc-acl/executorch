"""One native candidate: token checks, then the fixed seven-repeat six-cell session."""

import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
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
profile = sys.argv[1]
assert profile in ["4070ti-wmma-c08", "4070ti-wmma-c09", "4070ti-wmma-c10", "4070ti-wmma-c11", "4070ti-wmma-c06", "4070ti-wmma-c01", "4070ti-wmma-c02"]
assessment_path = artifacts / "stage" / ("s2-gate-" + profile.rsplit("-", 1)[-1]) / "candidate-assessment.json"
assessment = json.loads(assessment_path.read_text())
assert not assessment["numeric_failures"] and not assessment["command_failures"] and not assessment["non_dispatch_differences"]
assert not assessment["linear_not_ok"] and assessment["default_tiled_differences"] == 0
assert assessment["coverage"]["production_diff_cases"] > 0 and assessment["io_storage"] == "texture3d"
calibration_path = artifacts / "stage/s1-aa-mm-r1/calibration.json"
calibration = json.loads(calibration_path.read_text())
assert calibration["aa_within_limits"] and not calibration["historical_outside_3pct"]
repeats, clock_floor = calibration["repeats"], calibration["clock_floor_mhz"]
assert repeats == 7 and clock_floor == 2531
stage = artifacts / "stage" / ("s3-" + profile)
stage.mkdir(parents=True, exist_ok=True)
session_lock = (stage / "session.lock").open("a")
fcntl.flock(session_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
models = json.loads((artifacts / "preflight/models-static.json").read_text())
builds = {"control": "matrix-mm-r1", "candidate": "matrix-mm-r1"}
env = {k: v for k, v in os.environ.items() if not k.startswith(("ET_VK", "ETVK"))}
env.update(ET_VK_SARC_UNVERIFIED="1", ET_VK_SARC_DEV_PROFILE="4070ti-fused1",
           ETVK_DEVICE_INDEX=os.environ["WMMA_DEVICE_INDEX"], WMMA_PURPOSE="measurement",
           WMMA_SESSION=stage.name, WMMA_CANDIDATE=profile, PYTHONDONTWRITEBYTECODE="1",
           SARC_CAMPAIGN_TAG="wmma-nvidia-4070ti-20261015-" + stage.name, TMPDIR=str(artifacts / "tmp"))
assert not (artifacts / "GPU_GONE").exists()
assert not (artifacts / "stage/s0-topic-mm-control/ABORTED").exists()
control_gate = json.loads((artifacts / "stage/s0-topic-mm-control/snapshot-summary.json").read_text())
assert control_gate["observations"] == 37 and not control_gate["aborted"]
attention_control = json.loads((artifacts / "stage/s0-topic-mm-control/STAGE.json").read_text())["attention_control"]
assert attention_control == json.loads((assessment_path.parent / "STAGE.json").read_text())["attention_control"]
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
    if arm == "candidate":
        envs[arm]["ET_VK_SARC_WMMA_PROFILE"] = profile
    binaries[arm] = {"sha256": hashlib.sha256((arm_stage / "llama_main").read_bytes()).hexdigest(),
                     "commit": (arm_stage / "COMMIT").read_text().strip()}

observer = Observer(int(os.environ["WMMA_DEVICE_INDEX"]), os.getpid())
assert observer.identity["unit"] == "GPU-" + os.environ["WMMA_DEVICE_LOCK"]
device_lock = (Path.home() / ".cache/gpu-lab" / ("lock-" + os.environ["WMMA_DEVICE_LOCK"])).open("a")
rows_path = stage / "runs.csv"
fields = ["unit", "device_index", "driver", "session", "cell", "arm", "candidate", "profile", "commit", "binary_sha256", "wmma_profile",
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
        foreign = [c for c in sample["clients"] if c["own"] is not True]
        if foreign:
            with (stage / "foreign-waits.jsonl").open("a") as file:
                file.write(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "clients": foreign}) + "\n")
            time.sleep(60)
            deadline = time.monotonic() + maximum
            temperatures = []
            continue
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
            order = ["control", "candidate"] if rep % 2 else ["candidate", "control"]
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
                        save_row(dict(observer.identity, session=stage.name, cell=cell, arm=arm, candidate=profile, wmma_profile=profile if arm == "candidate" else "",
                                      profile="4070ti-fused1", commit=binaries[arm]["commit"], binary_sha256=binaries[arm]["sha256"],
                                      rep=rep, attempt=attempt, valid=0, reason="interrupted_before_csv_append",
                                      validity_mode="FIXED_CANDIDATE_SESSION", log=str((superseded / "runner.log").relative_to(stage)),
                                      record=str((superseded / "raw/record.json").relative_to(stage)),
                                      samples=str((superseded / "raw/samples.jsonl").relative_to(stage)),
                                      utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
                        continue
                    cool(target, 120)
                    child_env = dict(envs[arm], WMMA_RECORD_DIR=str(run_dir))
                    command = ["timeout", "--signal=TERM", "--kill-after=30", "1200", "python3",
                               str(stage / arm / "observer-tools/observe.py"), str(stage / arm / "llama_main"),
                               "--model_path", str(path), "--tokenizer_path", str(tokenizer),
                               "--prompt_file", str(artifacts / "stage/s0-parent-mm-verify/prompt_2048.txt"),
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
                        metrics = evaluate(log, run_dir / "samples.jsonl", result.returncode, clock_floor=clock_floor)
                    except Exception as error:
                        metrics = {"valid": 0, "reason": "evaluation_failed:" + str(error)}
                    if descendants:
                        metrics.update(valid=0, reason="sampling_or_runner_child_remains")
                    row = dict(observer.identity, session=stage.name, cell=cell, arm=arm, candidate=profile, wmma_profile=profile if arm == "candidate" else "",
                               profile="4070ti-fused1", commit=binaries[arm]["commit"], binary_sha256=binaries[arm]["sha256"],
                               rep=rep, attempt=attempt, rc=result.returncode, validity_mode="FIXED_CANDIDATE_SESSION",
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
                    if pair[0]["text_sha256"] != pair[1]["text_sha256"]:
                        timed_text_differences.append({"cell": cell, "rep": rep})
                    break
            else:
                raise RuntimeError(f"four attempts without valid candidate pair: {cell} r{rep}")


timed_text_differences = []
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
        result[cell] = {"arms": arms, "ratio": arms["candidate"]["median_tok_s"] / arms["control"]["median_tok_s"]}
    return result


def check_tokens(target):
    path = stage / "tokens.csv"
    token_fields = ["unit", "device_index", "driver", "session", "cell", "prompt", "arm", "candidate",
                    "profile", "wmma_profile", "commit", "binary_sha256", "text_sha256", "prompt_tokens",
                    "generated_tokens", "rc", "valid", "reason", "log", "record", "samples", "utc"]
    if path.exists():
        assert path.read_bytes().endswith(b"\n"), "partial token CSV; preserve and diagnose"
        with path.open() as file:
            tokens = list(csv.DictReader(file, strict=True))
    else:
        tokens = []
        with path.open("w", newline="") as file:
            csv.DictWriter(file, fieldnames=token_fields, lineterminator="\n").writeheader()
    prompts = {"timed": "prompt_2048.txt", "real": "prompt_real_2048.txt", "unaligned": "r1304.txt"}
    for model in models:
        model_path = Path(model["path"])
        cell = ("1b" if "-1b" in model_path.stem else "3b" if "-3b" in model_path.stem else "8b") + ("-8da4w" if "8da4w" in model_path.stem else "-4w")
        for prompt, filename in prompts.items():
            for arm in builds:
                if any(r["cell"] == cell and r["prompt"] == prompt and r["arm"] == arm and int(r["valid"]) for r in tokens):
                    continue
                hold_point(device_lock, artifacts)
                assert not (artifacts / "GPU_GONE").exists() and not (stage / arm / "ABORTED").exists()
                key = cell + "-" + prompt + "-" + arm
                raw = stage / "token-raw" / key
                log = stage / "token-logs" / (key + ".log")
                log.parent.mkdir(exist_ok=True)
                assert not raw.exists() and not log.exists(), "existing interrupted token attempt; diagnose before retry"
                cool(target, 120)
                command = ["timeout", "--signal=TERM", "--kill-after=30", "1200", "python3",
                           str(stage / arm / "observer-tools/observe.py"), str(stage / arm / "llama_main"),
                           "--model_path", str(model_path), "--tokenizer_path", str(model_path.parent.parent / "original/tokenizer.model"),
                           "--prompt_file", str(artifacts / "stage/s0-parent-mm-verify" / filename),
                           "--max_new_tokens", "1", "--temperature", "0"]
                if prompt == "timed":
                    command.append("--warmup")
                with log.open("w") as output:
                    result = subprocess.run(command, env=dict(envs[arm], WMMA_PURPOSE="next_token", WMMA_RECORD_DIR=str(raw)),
                                            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
                text = log.read_text(errors="replace")
                stats = [json.loads(line[line.index("{"):]) for line in text.splitlines() if "PyTorchObserver" in line]
                samples = [json.loads(line) for line in (raw / "samples.jsonl").read_text().splitlines()] if (raw / "samples.jsonl").exists() else []
                valid = result.returncode == 0 and len(stats) == 1 and stats[0]["generated_tokens"] == 0 and len(samples) >= 2
                valid = valid and all(not (s["event_reasons"] & 0x60) and all(c["own"] is True for c in s["clients"]) for s in samples)
                generated = "\n".join(line for line in text.splitlines() if "PyTorchObserver" not in line and not re.match(r"^[IWE] |^\[sarc_dev\]", line))
                row = dict(observer.identity, session=stage.name, cell=cell, prompt=prompt, arm=arm, candidate=profile,
                           profile="4070ti-fused1", wmma_profile=profile if arm == "candidate" else "",
                           commit=binaries[arm]["commit"], binary_sha256=binaries[arm]["sha256"],
                           text_sha256=hashlib.sha256(generated.encode()).hexdigest(),
                           prompt_tokens=stats[0]["prompt_tokens"] if stats else None,
                           generated_tokens=stats[0]["generated_tokens"] if stats else None,
                           rc=result.returncode, valid=int(valid), reason="" if valid else "token_run_failed_validity",
                           log=str(log.relative_to(stage)), record=str((raw / "record.json").relative_to(stage)),
                           samples=str((raw / "samples.jsonl").relative_to(stage)), utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                with path.open("a", newline="") as file:
                    csv.DictWriter(file, fieldnames=token_fields, lineterminator="\n").writerow(row)
                    file.flush()
                    os.fsync(file.fileno())
                tokens.append(row)
                if len(tokens) <= 3:
                    children = []
                    for entry in Path("/proc").iterdir():
                        if not entry.name.isdigit():
                            continue
                        try:
                            parent = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1])
                        except (FileNotFoundError, ProcessLookupError):
                            continue
                        if parent == os.getpid():
                            children.append(int(entry.name))
                    with (stage / "token-child-cleanup.jsonl").open("a") as file:
                        file.write(json.dumps({"run": key, "remaining_children": children}) + "\n")
                    assert not children, "child remains after token run; diagnose before continuing"
                assert valid and not (stage / arm / "ABORTED").exists(), "token run failed; retain and diagnose"
                print("token", cell, prompt, arm, row["text_sha256"], flush=True)
    comparisons = []
    for cell in cells:
        for prompt in prompts:
            pair = [r for r in tokens if r["cell"] == cell and r["prompt"] == prompt]
            assert len(pair) == 2 and {r["arm"] for r in pair} == set(builds)
            assert all(int(r["valid"]) == 1 for r in pair)
            same = pair[0]["text_sha256"] == pair[1]["text_sha256"] and int(pair[0]["prompt_tokens"]) == int(pair[1]["prompt_tokens"])
            comparisons.append({"cell": cell, "prompt": prompt, "same": same, "records": [r["record"] for r in pair]})
    summary = {"profile": profile, "runs": len(tokens), "comparisons": comparisons, "all_same": all(r["same"] for r in comparisons)}
    (stage / "token-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary["all_same"]


try:
    hold_point(device_lock, artifacts)
    if (stage / "STAGE.json").exists():
        manifest = json.loads((stage / "STAGE.json").read_text())
        assert manifest["binaries"] == binaries and manifest["candidate"] == profile
        target = manifest["cool_target_c"]
    else:
        idle = cool(50, 300)
        target = idle + 5
        dispatch = [r for r in assessment["production_diff_cases"] if r["case"].endswith("_Texture3D")]
        manifest = {"purpose": "native candidate session against same-session tuned control", "candidate": profile,
                    "binaries": binaries, "device": observer.identity, "cool_target_c": target,
                    "cool_rule": "initial <=50C/300s; then idle+5C/120s; stop after three 5s probes stop falling",
                    "protocol": {"repeats": repeats, "minimum_samples": 5, "clock_floor_mhz": clock_floor, "replacement_attempts": 4},
                    "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
                    "gate_assessment_sha256": hashlib.sha256(assessment_path.read_bytes()).hexdigest(),
                    "linear_dispatch_evidence": dispatch,
                    "dispatch_scope": "actual same-op production-shape microbench dispatch, not an instrumented model trace",
                    "attention_control": attention_control,
                    "orchestration_commit": os.environ["WMMA_ORCHESTRATION_COMMIT"]}
        (stage / "STAGE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tokens_same = check_tokens(target)
    run_rounds(1, repeats, target)
    final = medians(repeats)
    random_source = random.Random(407014)
    for cell, result in final.items():
        control = sorted(valid_rows(cell, "control", repeats), key=lambda r: int(r["rep"]))
        candidate = sorted(valid_rows(cell, "candidate", repeats), key=lambda r: int(r["rep"]))
        intervals = []
        for _ in range(20000):
            indices = [random_source.randrange(repeats) for _ in range(repeats)]
            intervals.append(statistics.median(float(candidate[i]["tok_s"]) for i in indices) /
                             statistics.median(float(control[i]["tok_s"]) for i in indices))
        intervals.sort()
        bounds = []
        for probability in [0.025, 0.975]:
            position = (len(intervals) - 1) * probability
            left, right = math.floor(position), math.ceil(position)
            bounds.append(intervals[left] + (intervals[right] - intervals[left]) * (position - left))
        result["ratio_interval95"] = bounds
    geomean = math.exp(statistics.mean(math.log(c["ratio"]) for c in final.values()))
    summary = {"profile": profile, "status": "SESSION_COMPLETE; ACTOR_AUDIT_AND_REVIEW_PENDING", "repeats": repeats,
               "correctness": "NEXT_TOKEN_EQUAL" if tokens_same else "CORRECTNESS_PENDING; top-2 logits unavailable from llama_main",
               "timed_text_differences": timed_text_differences,
               "clock_floor_mhz": clock_floor, "geomean_ratio": geomean, "cells": final,
               "valid_rows": sum(int(r["valid"]) for r in rows), "invalid_rows": sum(not int(r["valid"]) for r in rows),
               "within_member_numeric_margins": geomean >= 0.97 and all(c["ratio"] >= 0.92 for c in final.values()),
               "acceptance": "NONE; ONE_MEMBER_ONLY"}
    (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
except RuntimeError as error:
    if "NVML returned" in str(error):
        (artifacts / "GPU_GONE").write_text(str(error) + "\n")
    raise
finally:
    fcntl.flock(device_lock, fcntl.LOCK_UN)
    device_lock.close()
    observer.close()

