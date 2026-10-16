"""Record device-create chains from the four exact compiled runtime libraries."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from session_control import hold_point


tools = Path(__file__).resolve().parent
artifacts = Path(os.environ["WMMA_CAMPAIGN_ROOT"]) / ".artifacts"
checks = artifacts / "checks/enabled-features"
stage = artifacts / "stage/enabled-features-mm-r1"
assert not stage.exists(), "partial feature probe exists; diagnose before retry"
stage.mkdir()
lock = (Path.home() / ".cache/gpu-lab" / ("lock-" + os.environ["WMMA_DEVICE_LOCK"])).open("a")
rows = []
try:
    for tag in ["parent-mm-r1", "matrix-mm-r1"]:
        for component in ["llama", "backend"]:
            hold_point(lock, artifacts)
            source = checks / (tag + "-" + component)
            provenance = json.loads((source / "provenance.json").read_text())
            assert hashlib.sha256((source / "enabled_features").read_bytes()).hexdigest() == provenance["helper_binary_sha256"]
            assert hashlib.sha256((tools / "enabled_features.cpp").read_bytes()).hexdigest() == provenance["helper_source_sha256"]
            for name, expected in provenance["library_sha256"].items():
                assert hashlib.sha256((artifacts / "build" / tag / name).read_bytes()).hexdigest() == expected
            current = stage / (tag + "-" + component)
            current.mkdir()
            (current / "COMMIT").write_text(provenance["commit"] + "\n")
            shutil.copy2(source / "enabled_features", current)
            (current / "observer-tools").mkdir()
            for name in ["observe.py", "nvml_observer.py", "warm_file.py"]:
                shutil.copy2(tools / name, current / "observer-tools" / name)
            env = {k: v for k, v in os.environ.items() if not k.startswith(("ET_VK", "ETVK"))}
            env.update(WMMA_STAGE=str(current), WMMA_PURPOSE="enabled_feature_probe",
                       ETVK_DEVICE_INDEX=os.environ["WMMA_DEVICE_INDEX"],
                       ET_VK_SARC_UNVERIFIED="1", ET_VK_SARC_DEV_PROFILE="4070ti-fused1")
            with (current / "output.log").open("w") as output:
                result = subprocess.run(["timeout", "--signal=TERM", "--kill-after=30", "120", "python3",
                                         str(current / "observer-tools/observe.py"), str(current / "enabled_features")],
                                        env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
            assert result.returncode == 0, "feature probe failed; preserve and diagnose"
            outputs = [json.loads(line) for line in (current / "output.log").read_text().splitlines() if line.startswith("{")]
            assert len(outputs) == 1
            features = outputs[0]
            assert features == {"extension_enabled": True, "vulkanMemoryModel": 1,
                                "vulkanMemoryModelDeviceScope": 0,
                                "vulkanMemoryModelAvailabilityVisibilityChains": 0,
                                "vkCreateDevice_result": 0}
            records = list((current / "observations").glob("*/record.json"))
            assert len(records) == 1 and json.loads(records[0].read_text())["returncode"] == 0
            rows.append(dict(provenance, enabled_features=features, probe_returncode=result.returncode))
            print(tag, component, features, flush=True)
            fcntl.flock(lock, fcntl.LOCK_UN)
finally:
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
(checks / "summary.json").write_text(json.dumps({"probes": rows, "scope": "exact static runtime libraries; no model, shader, dispatch or timed-run instrumentation"}, indent=2) + "\n")
