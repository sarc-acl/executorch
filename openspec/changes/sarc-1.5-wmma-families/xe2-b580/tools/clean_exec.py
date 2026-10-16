#!/usr/bin/env python3
import os
from pathlib import Path
import sys

env = {key: value for key, value in os.environ.items()
       if not key.startswith(("ET_VK_", "ETVK_", "VK_"))}
for line in Path(sys.argv[1]).read_text().splitlines():
    key, value = line.split("=", 1)
    assert key in ("ET_VK_SARC_UNVERIFIED", "ET_VK_SARC_DEV_PROFILE"), key
    env[key] = value
env["ETVK_DEVICE_INDEX"] = "0"
env["VK_DRIVER_FILES"] = "/usr/share/vulkan/icd.d/intel_icd.x86_64.json"
os.execvpe(sys.argv[2], sys.argv[2:], env)
