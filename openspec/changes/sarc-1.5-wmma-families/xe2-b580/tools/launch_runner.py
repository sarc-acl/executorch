#!/usr/bin/env python3
"""Prime a staged model and exec the recorded binary with explicit controls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

stage = Path(sys.argv[0]).resolve().parent
args = sys.argv[1:]
model = args[args.index("--model_path") + 1]
root = Path(os.environ["B580_ARTIFACTS"])
if (root / "DEVICE_LOST").exists():
    sys.exit(77)
env = {key: value for key, value in os.environ.items()
       if not key.startswith(("ET_VK_", "ETVK_", "VK_"))}
for line in (stage / "env").read_text().splitlines():
    key, value = line.split("=", 1)
    env[key] = value
if os.environ.get("ET_VK_FORCE_TILED_LINEAR") == "1":
    env["ET_VK_FORCE_TILED_LINEAR"] = "1"
env["ETVK_DEVICE_INDEX"] = "0"
env["VK_DRIVER_FILES"] = "/usr/share/vulkan/icd.d/intel_icd.x86_64.json"
start = time.monotonic()
subprocess.run(["cat", model], stdout=subprocess.DEVNULL, check=True)
with (stage / "page-cache.jsonl").open("a") as file:
    file.write(json.dumps({"pid": os.getpid(), "model": model, "primed": True,
                           "read_seconds": time.monotonic() - start,
                           "epoch_ns": time.time_ns()}) + "\n")
os.execve(stage / "llama_main.bin", [str(stage / "llama_main.bin"), *args], env)
