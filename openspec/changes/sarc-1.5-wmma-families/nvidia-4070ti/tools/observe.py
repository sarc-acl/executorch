"""Observe one unmodified gate command, retaining status, sensors and clients."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from nvml_observer import Observer


stage = Path(os.environ["WMMA_STAGE"])
artifacts = Path(os.environ["WMMA_CAMPAIGN_ROOT"]) / ".artifacts"
binary = Path(sys.argv[1])
arguments = sys.argv[2:]
record_dir = stage / "observations" / str(time.time_ns())
record_dir.mkdir(parents=True)
record = {"purpose": "parent_snapshot", "binary": str(binary), "argv": arguments,
          "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
          "commit": (stage / "COMMIT").read_text().strip(),
          "profile": os.environ.get("ET_VK_SARC_DEV_PROFILE"),
          "candidate": "control", "session": stage.name,
          "environment": {k: v for k, v in os.environ.items() if k.startswith(("ET_VK", "ETVK"))},
          "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "timing_validity": "NOT_A_TIMED_SESSION", "reasons": [],
          "sensor_evidence": str(record_dir / "samples.jsonl")}
child, observer = None, None
returncode = 77


def interrupted(signum, _frame):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signum)
    raise SystemExit(f"received signal {signum}")


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    if (artifacts / "GPU_GONE").exists() or (stage / "ABORTED").exists():
        raise RuntimeError("an earlier observation blocks later GPU commands")
    if "--model_path" in arguments:
        model = Path(arguments[arguments.index("--model_path") + 1])
        record["model"] = str(model)
        record["cell"] = model.stem
        last = stage / ".last-model"
        changed = not last.exists() or last.read_text() != str(model)
        cache = subprocess.check_output(["fincore", "-n", "-b", "-o", "RES,SIZE", str(model)], text=True)
        resident, size = map(int, cache.split())
        record.update(cache_before=min(1.0, resident / size), resident_before=resident, model_bytes=size)
        passes = 0
        while passes < 3 and (changed or resident < size):
            subprocess.run(["python3", str(Path(__file__).with_name("warm_file.py")), str(model)], check=True)
            passes += 1
            changed = False
            resident, size = map(int, subprocess.check_output(["fincore", "-n", "-b", "-o", "RES,SIZE", str(model)], text=True).split())
        last.write_text(str(model))
        record.update(cache_after=min(1.0, resident / size), resident_after=resident, warm_passes=passes, slow_load_possible=resident < size)
    observer = Observer(int(os.environ["WMMA_DEVICE_INDEX"]), os.getpid())
    record.update(observer.identity)
    with (record_dir / "samples.jsonl").open("w") as samples:
        def sample():
            row = observer.sample()
            samples.write(json.dumps(row) + "\n")
            if any(client["own"] is not True for client in row["clients"]):
                raise RuntimeError("foreign or unattributed GPU client observed")

        sample()
        child = subprocess.Popen([str(binary), *arguments], start_new_session=True)
        while child.poll() is None:
            sample()
            time.sleep(0.005)
        returncode = child.wait()
        sample()
        if "--model_path" in arguments and returncode != 0:
            record["reasons"].append("model runner failed; diagnose before any retry")
            (stage / "ABORTED").write_text(f"model runner rc={returncode}; see {record_dir}\n")
except BaseException as error:
    returncode = 77
    record["reasons"].append(str(error))
    if not (stage / "ABORTED").exists():
        (stage / "ABORTED").write_text(str(error) + "\n")
    if "NVML returned" in str(error):
        (artifacts / "GPU_GONE").write_text(str(error) + "\n")
        status = artifacts.parent / "executorch/openspec/changes/sarc-1.5-wmma-families/nvidia-4070ti/STATUS.md"
        with status.open("a") as output:
            output.write("\n## DEVICE LOST\n\nNVML query failed; later GPU commands are blocked by the artifact GPU_GONE marker. Owner recovery required; nothing retried.\n")
        returncode = 70
    elif "GPU client" in str(error):
        returncode = 76
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
finally:
    if observer is not None:
        try:
            observer.close()
        except RuntimeError as error:
            record["reasons"].append(str(error))
            returncode = 70
    record.update(returncode=returncode, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (record_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n")
sys.exit(128 - returncode if returncode < 0 else returncode)
