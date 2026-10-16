"""NVIDIA family finalists (coordinator decision 2026-10-16 08:20 UTC): one rebuild, control check, full gates, sessions."""

import datetime
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
from session_control import hold_point


root = Path(os.environ["WMMA_CAMPAIGN_ROOT"])
artifacts = root / ".artifacts"
change = root / "executorch/openspec/changes/sarc-1.5-wmma-families/nvidia-4070ti"
tools = Path(__file__).resolve().parent
commit = sys.argv[1]
tag, reference_tag = "matrix-fam-r1", "matrix-mm-r1"
build = artifacts / "build" / tag
manifest = root / "inputs/inventory-f393c3ff7f/manifests/nvidia-ampere-ada.json"
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == "ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1"
suffix = "_texture3d_texture2d_half"
finalists = {
    "4070ti-nv-family-a": ["sarc_linear_q4gsw_coopmat_orin_t256x128k16g42s32bt" + suffix,
                           "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf_t128x128k64g24s32mk32ra" + suffix],
    "4070ti-nv-family-b": ["sarc_dev_linear_q4gsw_coopmat_bx_t128x128k32g42s32f32c" + suffix,
                           "sarc_linear_q4gsw_coopmat_orin_t256x128k16g42s32bt" + suffix,
                           "sarc_linear_dq8ca_coopmat_zpgtr_orin_bf_t128x128k64g24s32mk32ra" + suffix],
}
sdpa_cases = {"extended": 8, "full": 4}
summary_path = change / "results/family-finalists.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}


def save(note):
    summary["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    with (change / "STATUS.md").open("a") as file:
        file.write("\n" + summary["updated_utc"] + " family finalists: " + note + "\n")
    print(note, flush=True)


def normalized(text):
    return re.sub(r"(prefill_tok_s|decode_tok_s)=[0-9.]*", r"\1=<RATE>", text)


def file_hashes(path):
    return dict(reversed(line.split(maxsplit=1)) for line in path.read_text().splitlines())


assert not (artifacts / "GPU_GONE").exists()
if "build" not in summary:
    assert not build.exists(), "partial finalist build exists; diagnose before any rebuild"
    subprocess.run([str(tools / "build-parent.sh"), tag, commit], check=True)
    assert (build / "COMMIT").read_text().strip() == commit
    for name in ["golden-llama.txt", "golden-backend.txt"]:
        assert (build / name).read_text().strip() == "spirv golden: PASS (53 shipped variants)"
    with (build / "module-shared.json").open("w") as output:
        subprocess.run(["python3", str(tools / "module_shared.py"), str(manifest),
                        str(build / "llama/vulkan_compute_shaders")], stdout=output, check=True)
    comparisons = []
    for component in ["llama", "backend"]:
        reference = artifacts / "build" / reference_tag / component / "vulkan_compute_shaders"
        current = build / component / "vulkan_compute_shaders"
        paths = sorted(reference.glob("*.spv"))
        assert {p.name for p in paths} == {p.name for p in current.glob("*.spv")}
        assert all(path.read_bytes() == (current / path.name).read_bytes() for path in paths)
        comparisons.append({"component": component, "unchanged_spirv_count": len(paths)})
    (build / "unchanged-spirv.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    before = file_hashes(artifacts / "build" / reference_tag / "source-files.sha256")
    after = file_hashes(build / "source-files.sha256")
    changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
    runtime_changed = [p for p in changed if not p.startswith("./openspec/changes/sarc-1.5-wmma-families/nvidia-4070ti/")]
    assert runtime_changed == ["./backends/vulkan/runtime/graph/ops/impl/sarc_dev/NvidiaWmmaMatrix.cpp"], runtime_changed
    (build / "source-diff.json").write_text(json.dumps({"reference": reference_tag, "changed": changed}, indent=2) + "\n")
    summary["build"] = {"tag": tag, "commit": commit, "binaries": (build / "binaries.sha256").read_text().splitlines(),
                        "golden": "PASS (53 shipped variants), llama and backend",
                        "spirv_unchanged_vs": reference_tag, "spirv": comparisons,
                        "source_changed_vs_reference": changed,
                        "memory_model_probe": reference_tag + " (runtime source identical except the dev-zone selector)"}
    save("build " + tag + " of " + commit + " complete; golden PASS, SPIR-V identical to " + reference_tag + ".")

env = dict(os.environ, WMMA_PROBE_TAG=reference_tag)
if "control" not in summary:
    stage = artifacts / "stage/s0-topic-fam-control"
    subprocess.run(["python3", str(tools / "parent_snapshot.py"), tag, stage.name], env=env, check=True)
    reference = normalized((artifacts / "stage/s0-topic-mm-control/verify.out").read_text())
    diff = "".join(difflib.unified_diff(reference.splitlines(True), normalized((stage / "verify.out").read_text()).splitlines(True),
                                        fromfile="s0-topic-mm-control", tofile=stage.name))
    (stage / "control-output.diff").write_text(diff)
    summary["control"] = {"stage": str(stage.relative_to(artifacts)), "identical_to_matrix_control": not diff}
    save("control verify on " + tag + ": " + ("identical to s0-topic-mm-control (rates removed)" if not diff else "DIFFERS; see control-output.diff"))
    assert not diff, "control dispatch or output changed on the rebuild; diagnose"

device_lock = (Path.home() / ".cache/gpu-lab" / ("lock-" + os.environ["WMMA_DEVICE_LOCK"])).open("a")
gates = summary.setdefault("gates", {})
for profile, shaders in finalists.items():
    if profile in gates:
        continue
    stage = artifacts / "stage" / ("s2-gate-" + profile.removeprefix("4070ti-"))
    assert not stage.exists(), "partial finalist gate exists; diagnose before any retry"
    gate_env = dict(env, WMMA_MATRIX_PROFILE=profile, WMMA_CANDIDATE=profile, WMMA_PURPOSE="candidate_gate", WMMA_SESSION=stage.name)
    subprocess.run(["python3", str(tools / "parent_snapshot.py"), tag, stage.name], env=gate_env, check=True)
    records = [json.loads(p.read_text()) for p in sorted((stage / "observations").glob("*/record.json"))]
    assert len(records) == 37 and all(not r["reasons"] for r in records)
    assert all(r["environment"].get("ET_VK_SARC_WMMA_PROFILE") == profile for r in records)
    microbench = [r for r in records if "model" not in r]
    correctness = next(r for r in microbench if "--correctness-only" in r["argv"])
    pdiff_records = [r for r in microbench if "--production-diff" in r["argv"]]
    pdiff_cases, numeric_failures = [], []
    for path in sorted((stage / "verify").glob("pdiff-*.log")):
        text = path.read_text()
        for match in re.finditer(r"^\[production-diff\] (.*?) -> (\S+) .*?, correctness=(.*)$", text, re.MULTILINE):
            pdiff_cases.append({"log": str(path.relative_to(stage)), "case": match[1], "shader": match[2], "correctness": match[3]})
        if "[production-diff] ALL PASSED" not in text:
            numeric_failures.append(str(path.relative_to(stage)))
    if correctness["returncode"] != 0:
        numeric_failures.append("verify/correctness.log")
    command_failures = [{"argv": r["argv"], "returncode": r["returncode"]} for r in [correctness, *pdiff_records] if r["returncode"] != 0]
    linear = []
    for scheme in ["4w", "8da4w"]:
        linear.extend(json.loads((stage / "verify" / ("linear-" + scheme + ".json")).read_text())["cases"])
    linear_not_ok = [row for row in linear if not row["ok"]]
    reference = (artifacts / "stage/s0-topic-mm-control/verify.out").read_text()
    output = (stage / "verify.out").read_text()
    (stage / "control-output.diff").write_text("".join(difflib.unified_diff(
        normalized(reference).splitlines(True), normalized(output).splitlines(True), fromfile="control", tofile=profile)))
    clean_reference = [line.split(" kernels:")[0] if line.startswith("linear ") else line for line in normalized(reference).splitlines()]
    clean_current = [line.split(" kernels:")[0] if line.startswith("linear ") else line for line in normalized(output).splitlines()]
    non_dispatch_differences = [{"control": a, "candidate": b} for a, b in zip(clean_reference, clean_current) if a != b]
    if len(clean_reference) != len(clean_current):
        non_dispatch_differences.append({"control_lines": len(clean_reference), "candidate_lines": len(clean_current)})
    coverage = {shader: {"linear_benchmark_cases": sum(r["kernel"] == shader for r in linear),
                         "production_diff_cases": sum(r["shader"] == shader for r in pdiff_cases)} for shader in shaders}

    sdpa = stage / "sdpa"
    sdpa.mkdir()
    sdpa_env = {k: v for k, v in os.environ.items() if not k.startswith(("ET_VK", "ETVK"))}
    sdpa_env.update(ET_VK_SARC_UNVERIFIED="1", ET_VK_SARC_DEV_PROFILE="4070ti-fused1", ETVK_DEVICE_INDEX=os.environ["WMMA_DEVICE_INDEX"],
                    WMMA_STAGE=str(stage), WMMA_PURPOSE="sdpa_correctness", WMMA_SESSION=stage.name, TMPDIR=str(artifacts / "tmp"),
                    SARC_CAMPAIGN_TAG="wmma-nvidia-4070ti-20261015-" + stage.name)
    sdpa_failures, sdpa_kernels = [], {}
    passes = [("candidate", tier, i) for tier in sdpa_cases for i in range(1, 13)] + [("control", tier, 1) for tier in sdpa_cases]
    for arm, tier, i in passes:
        hold_point(device_lock, artifacts)
        assert not (artifacts / "GPU_GONE").exists() and not (stage / "ABORTED").exists()
        key = f"{arm}-{tier}-r{i}"
        run_env = dict(sdpa_env, WMMA_RECORD_DIR=str(sdpa / "raw" / key), WMMA_CANDIDATE=profile if arm == "candidate" else "control")
        if arm == "candidate":
            run_env["ET_VK_SARC_WMMA_PROFILE"] = profile
        with (sdpa / (key + ".log")).open("w") as log:
            result = subprocess.run(["timeout", "--signal=TERM", "--kill-after=30", "1800", "python3",
                                     str(stage / "observer-tools/observe.py"), str(stage / "bin/test_llama_microbench"),
                                     "--sdpa-correctness-only", "--sdpa-tier=" + tier],
                                    env=run_env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        fcntl.flock(device_lock, fcntl.LOCK_UN)
        lines = (sdpa / (key + ".log")).read_text(errors="replace").splitlines()
        cases = [l for l in lines if l.startswith("[sdpa-correctness]") and " S=" in l]
        totals = [l for l in lines if l.startswith(f"[sdpa-correctness] tier={tier} ")]
        kernels = [l for l in lines if l.startswith("[sdpa-kernels]")]
        problems = []
        if result.returncode != 0:
            problems.append(f"rc={result.returncode}")
        if len(totals) != 1 or f"cases_run={sdpa_cases[tier]}" not in totals[0].split():
            problems.append(f"summary {totals}")
        if len(cases) != sdpa_cases[tier] or len(kernels) != sdpa_cases[tier]:
            problems.append(f"{len(cases)} cases / {len(kernels)} kernel lines")
        problems += [l for l in cases if not (l.rstrip().endswith(" PASSED") and re.search(r" mismatches=0/\d+", l))]
        problems += [l for l in kernels if not l.rstrip().endswith("pairing=ok")]
        if problems:
            sdpa_failures.append({"pass": key, "problems": problems})
        sdpa_kernels.setdefault(arm + "-" + tier, set()).add(tuple(kernels))
        assert not (artifacts / "GPU_GONE").exists() and not (stage / "ABORTED").exists(), "observer failure; diagnose"
    kernel_sets = {k: [list(v) for v in s] for k, s in sdpa_kernels.items()}
    attention_identical = all(sdpa_kernels["candidate-" + t] == sdpa_kernels["control-" + t] and len(sdpa_kernels["control-" + t]) == 1
                              for t in sdpa_cases)

    verdict = "GATE_CAPTURE_COMPLETE; TOKENS_AND_SESSION_PENDING"
    if not all(c["production_diff_cases"] for c in coverage.values()):
        verdict = "MISSING_PRODUCTION_DIFF_DISPATCH; DIAGNOSIS_PENDING"
    if non_dispatch_differences or linear_not_ok or not attention_identical:
        verdict = "CONTROL_DIFFERENCES_REQUIRE_DIAGNOSIS; NO_ACCEPTANCE"
    if numeric_failures or command_failures or sdpa_failures:
        verdict = "CORRECTNESS_GATE_FAILED; NO_RETRY_OR_ACCEPTANCE"
    assessment = {"profile": profile, "shaders": shaders, "scheme": "4w+8da4w", "io_storage": "texture3d", "weight_storage": "texture2d",
                  "stage": str(stage.relative_to(artifacts)), "commit": (stage / "COMMIT").read_text().strip(), "verdict": verdict,
                  "coverage": {"production_diff_cases": sum(c["production_diff_cases"] for c in coverage.values()), "per_shader": coverage},
                  "numeric_failures": numeric_failures, "command_failures": command_failures, "linear_not_ok": linear_not_ok,
                  "non_dispatch_differences": non_dispatch_differences, "default_tiled_differences": output.count("output DIFFER"),
                  "production_diff_cases": pdiff_cases, "linear_cases": linear,
                  "sdpa": {"passes": len(passes), "failures": sdpa_failures, "kernel_lines": kernel_sets,
                           "attention_identical_to_control": attention_identical},
                  "timing": "NOT_RUN", "acceptance": "NONE"}
    (stage / "candidate-assessment.json").write_text(json.dumps(assessment, indent=2) + "\n")
    gates[profile] = {k: assessment[k] for k in ["stage", "commit", "verdict", "coverage", "numeric_failures", "non_dispatch_differences",
                                                 "default_tiled_differences"]}
    gates[profile]["sdpa"] = {"passes": len(passes), "failures": len(sdpa_failures), "attention_identical_to_control": attention_identical}
    save(profile + " gate: " + verdict + "; coverage " + json.dumps({s.removesuffix(suffix): c["production_diff_cases"] for s, c in coverage.items()}) +
         "; SDPA extended/full 12 passes each, failures " + str(len(sdpa_failures)) + ". Assessment " + str((stage / "candidate-assessment.json").relative_to(artifacts)) + ".")
    assert verdict.startswith("GATE_CAPTURE_COMPLETE"), "finalist gate needs diagnosis before timing"
device_lock.close()

sessions = summary.setdefault("sessions", {})
for profile in finalists:
    if profile in sessions:
        continue
    stage = artifacts / "stage" / ("s3-" + profile)
    if not (stage / "summary.json").exists():
        subprocess.run(["timeout", "--signal=TERM", "--kill-after=60", "12h", "python3", str(tools / "candidate_session.py"), profile],
                       env=dict(os.environ, WMMA_BUILD=tag), check=True)
    result = json.loads((stage / "summary.json").read_text())
    sessions[profile] = dict(result, stage=str(stage.relative_to(artifacts)))
    save(profile + " session: " + result["status"] + "; correctness=" + result["correctness"] + "; ratios " +
         json.dumps({c: round(r["ratio"], 4) for c, r in result["cells"].items()}) + "; geomean=" + str(result["geomean_ratio"]) + ".")
print("FAMILY_FINALISTS_COMPLETE; review pending", flush=True)
