"""Capture the unmodified six-cell parent verify output, without accepting rows."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from nvml_observer import Observer


root = Path(os.environ["WMMA_CAMPAIGN_ROOT"])
artifacts = root / ".artifacts"
tools = Path(__file__).resolve().parent
build = artifacts / "build" / sys.argv[1]
tree = artifacts / "src" / sys.argv[1] / "executorch"
stage = artifacts / "stage" / (sys.argv[2] if len(sys.argv) > 2 else "s0-parent-mm-verify")
assert not stage.exists(), "snapshot exists; preserve it and diagnose rather than overwrite"
assert not (artifacts / "GPU_GONE").exists()
source_commit = (build / "COMMIT").read_text().strip()
subprocess.run(["git", "-C", str(root / "executorch"), "merge-base", "--is-ancestor",
                "617897184529ff180af491ee0d5d7de778cbb6c6", source_commit], check=True)
for name in ["golden-llama.txt", "golden-backend.txt"]:
    assert (build / name).read_text().strip() == "spirv golden: PASS (53 shipped variants)"
metadata = json.loads((artifacts / "metadata/parent-mm-r1/runtime-results.json").read_text())
assert len(metadata) == 6 and all(row["matches_static"] for row in metadata)
probes = json.loads((artifacts / "checks/enabled-features/summary.json").read_text())["probes"]
# A rebuild that changes only dev-zone selectors reuses the probe of its runtime-identical build (WMMA_PROBE_TAG).
probe_tag = os.environ.get("WMMA_PROBE_TAG", sys.argv[1])
probe_commit = (artifacts / "build" / probe_tag / "COMMIT").read_text().strip()
matching = [row for row in probes if row["tag"] == probe_tag and row["commit"] == probe_commit]
assert len(matching) == 2 and all(row["enabled_features"]["vulkanMemoryModel"] == 1 for row in matching)
modules = json.loads((build / "module-shared.json").read_text())
assert len(modules["modules"]) == 11 and all(row["matches_source_declarations"] for row in modules["modules"])
formats = json.loads((artifacts / "preflight/formats.json").read_text())
assert len(formats["formats"]) == 6

held = False
while (artifacts / "HOLD").exists():
    (artifacts / "HELD").write_text("HELD before parent verify snapshot\n")
    held = True
    time.sleep(60)
if held:
    (artifacts / "HELD").unlink()

stage.mkdir(parents=True)
(stage / "bin").mkdir()
(stage / "observer-tools").mkdir()
for name in ["observe.py", "nvml_observer.py", "warm_file.py"]:
    shutil.copy2(tools / name, stage / "observer-tools" / name)
sources = {
    "llama_main": build / "llama/examples/models/llama/llama_main",
    "test_llama_microbench": build / "tests/test_llama_microbench",
}
for name, source in sources.items():
    shutil.copy2(source, stage / "bin" / name)
    wrapper = stage / name
    wrapper.write_text('#!/bin/bash\nexec python3 "$WMMA_STAGE/observer-tools/observe.py" "$WMMA_STAGE/bin/' + name + '" "$@"\n')
    wrapper.chmod(0o755)
shutil.copy2(build / "llama/examples/models/llama/runner/libllama_runner.so", stage)
shutil.copy2(build / "COMMIT", stage)
kit = tree / "openspec/changes/sarc-1.5-e2e-benchmark/kit/prompts"
for prompt in ["prompt_2048.txt", "prompt_check.txt", "prompt_real_2048.txt"]:
    shutil.copy2(kit / prompt, stage)
shutil.copy2(tree / "openspec/changes/sarc-1.5-4070ti-fused-port/tools/r1304.txt", stage)
env = {key: value for key, value in os.environ.items() if not key.startswith(("ET_VK", "ETVK"))}
env.update(ET_VK_SARC_UNVERIFIED="1", ET_VK_SARC_DEV_PROFILE="4070ti-fused1",
           WMMA_STAGE=str(stage), TMPDIR=str(artifacts / "tmp"),
           SARC_CAMPAIGN_TAG="wmma-nvidia-4070ti-20261015-" + stage.name)
if os.environ.get("WMMA_MATRIX_PROFILE"):
    env["ET_VK_SARC_WMMA_PROFILE"] = os.environ["WMMA_MATRIX_PROFILE"]
manifest = {"orchestration_commit": os.environ.get("WMMA_ORCHESTRATION_COMMIT") or subprocess.check_output(["git", "-C", str(root / "executorch"), "rev-parse", "HEAD"], text=True).strip(),
            "source_commit": (build / "COMMIT").read_text().strip(), "build": sys.argv[1],
            "environment": {k: v for k, v in env.items() if k.startswith(("ET_VK", "ETVK"))},
            "attention_control": {"fallback": "k4070tiRefine1", "softmax": "4070ti_nzf",
                                  "fused": ["fused3sb_d64_t32x32g11s32rko", "fused3sb_d128_t16x64g11s32rko"]},
            "purpose": "snapshot only; no starting-row acceptance or timed-session validity"}
if os.environ.get("WMMA_MATRIX_PROFILE"):
    manifest["purpose"] = "candidate gate capture; dispatch and correctness assessment saved separately"
manifest["hashes"] = {str(path.relative_to(stage)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in stage.rglob("*") if path.is_file()}
manifest["verify_sha256"] = hashlib.sha256((tree / "sarc/tools/verify.sh").read_bytes()).hexdigest()
(stage / "STAGE.json").write_text(json.dumps(manifest, indent=2) + "\n")

observer = Observer(int(os.environ["WMMA_DEVICE_INDEX"]), os.getpid())
try:
    assert observer.identity["unit"] == "GPU-" + os.environ["WMMA_DEVICE_LOCK"]
    sample = observer.sample()
    assert not sample["clients"], "foreign GPU clients before snapshot"
    temperatures = []
    for _ in range(61):
        sample = observer.sample()
        temperatures.append(sample["temperature_c"])
        if sample["temperature_c"] <= 50 or (len(temperatures) >= 4 and temperatures[-1] >= temperatures[-4]):
            break
        time.sleep(5)
finally:
    observer.close()
command = [str(tree / "sarc/tools/verify.sh"), "--dir", str(stage), "--lock", os.environ["WMMA_DEVICE_LOCK"],
           "--device-index", os.environ["WMMA_DEVICE_INDEX"], "--model-root", os.environ["WMMA_MODEL_ROOT"],
           "--models", "1b,3b,8b", "--schemes", "4w,8da4w", "--pdiff"]
with (stage / "verify.out").open("w") as output:
    result = subprocess.run(["timeout", "--signal=TERM", "--kill-after=60", "2h", *command], env=env,
                            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
records = [json.loads(path.read_text()) for path in sorted((stage / "observations").glob("*/record.json"))]
summary = {"verify_returncode": result.returncode, "observations": len(records),
           "runner_observations": sum("model" in record for record in records),
           "aborted": (stage / "ABORTED").exists(), "purpose": "parent snapshot; not a candidate pass"}
if os.environ.get("WMMA_MATRIX_PROFILE"):
    summary["purpose"] = "candidate gate capture; not an acceptance verdict"
(stage / "snapshot-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
assert result.returncode == 0 and not summary["aborted"] and len(records) == 37 and summary["runner_observations"] == 22
