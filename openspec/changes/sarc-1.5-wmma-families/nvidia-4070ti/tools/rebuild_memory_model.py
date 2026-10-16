"""Build both approved memory-model arms sequentially, retaining exact evidence."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


tools = Path(__file__).resolve().parent
root = Path(os.environ["WMMA_CAMPAIGN_ROOT"])
artifacts = root / ".artifacts"
control_commit, selector_commit = sys.argv[1:]
manifest = root / "inputs/inventory-f393c3ff7f/manifests/nvidia-ampere-ada.json"
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == "ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1"


def retire_objects(tag):
    build = artifacts / "build" / tag
    subprocess.run(["sha256sum", "-c", str(build / "binaries.sha256")], check=True)
    directories = sorted(build.rglob("CMakeFiles"))
    directories = [p for p in directories if not any(parent.name == "CMakeFiles" for parent in p.relative_to(build).parents)]
    archive = build / "retired-cmake-provenance.tar.gz"
    assert not archive.exists(), "retirement evidence already exists; inspect instead of overwriting"
    records = []
    with tarfile.open(archive, "x:gz") as output:
        for directory in directories:
            for path in sorted(directory.rglob("*")):
                if not path.is_file():
                    continue
                assert path.suffix not in [".spv", ".so", ".a"]
                records.append({"path": str(path.relative_to(build)), "bytes": path.stat().st_size,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                if not path.name.endswith((".o", ".o.d")):
                    output.add(path, arcname=str(path.relative_to(build)))
    (build / "retired-objects.json").write_text(json.dumps(records, indent=2) + "\n")
    for directory in directories:
        shutil.rmtree(directory)
    subprocess.run(["sha256sum", "-c", str(build / "binaries.sha256")], check=True)
    print("retired superseded intermediate objects:", tag, flush=True)


retire_objects("matrix-r1")
for tag, commit in [("parent-mm-r1", control_commit), ("matrix-mm-r1", selector_commit)]:
    subprocess.run([str(tools / "build-parent.sh"), tag, commit], check=True)
    build = artifacts / "build" / tag
    with (build / "module-shared.json").open("w") as output:
        subprocess.run(["python3", str(tools / "module_shared.py"), str(manifest),
                        str(build / "llama/vulkan_compute_shaders")], stdout=output, check=True)
    comparisons = []
    for component in ["llama", "backend"]:
        reference = artifacts / "build/parent-r1" / component / "vulkan_compute_shaders"
        current = build / component / "vulkan_compute_shaders"
        paths = sorted(reference.glob("*.spv"))
        assert {p.name for p in paths} == {p.name for p in current.glob("*.spv")}
        assert all(path.read_bytes() == (current / path.name).read_bytes() for path in paths)
        comparisons.append({"component": component, "unchanged_spirv_count": len(paths)})
    (build / "unchanged-spirv.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    if tag == "parent-mm-r1":
        retire_objects(tag)
print("BOTH_FEATURE_ENABLED_SOURCE_BUILDS_COMPLETE; no model or GPU execution launched", flush=True)
