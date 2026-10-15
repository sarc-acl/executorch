#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
TAG=$1
[[ $TAG =~ ^parent-[a-zA-Z0-9-]+$ ]] || exit 2
P=$A/build/$TAG
T=$A/src/$TAG/executorch
[[ ! -e $P && ! -e $A/src/$TAG ]] || { echo "immutable build tag already exists" >&2; exit 2; }
mkdir -p "$A/build" "$A/src" "$A/bin" "$A/docker-config"
take_lock
hold_point "parent build $TAG"
no_others "before parent build"
gone_check "before parent build"
python3 - "$A" <<'PY'
import shutil, sys
free = shutil.disk_usage(sys.argv[1]).free
print(f"free bytes before export/build: {free}", flush=True)
if free < 32 * 1024**3:
    raise SystemExit("less than 32 GiB free; stop before exporting/building")
PY
mkdir "$P"
exec > >(tee "$P/provenance.txt") 2>&1
printf 'tag %s\nparent %s\nstarted %s\n' "$TAG" "$PARENT_COMMIT" "$(date -u +%FT%TZ)"
git -C "$ET" rev-parse HEAD > "$P/orchestration-commit.txt"
sha256sum "$TOOLS/build-parent.sh" "$TOOLS/mktree.sh" "$TOOLS/common.sh" "$TOOLS/podman-shim.sh"
"$TOOLS/mktree.sh" "$PARENT_COMMIT" "$T"
cp "$A/src/$TAG/COMMIT" "$A/src/$TAG/SUBMODULES" "$P/"
find "$T" -type d -exec chmod u+w {} +
(
  cd "$T"
  find . -type f -not -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum
) > "$P/source-files.sha256"
sha256sum "$P/source-files.sha256" "$P/COMMIT" "$P/SUBMODULES"
install -m 755 "$TOOLS/podman-shim.sh" "$A/bin/podman"
export PATH=$A/bin:$PATH DOCKER_CONFIG=$A/docker-config
export SARC_BUILD_IMAGE=sha256:f7d5b7f0eeea72b33b3a36efdb79307ccd1b4bded375451f330f52dfd4f1cb02
export SARC_JOBS=4
docker image inspect "$SARC_BUILD_IMAGE" > "$P/image.json"
docker run --rm --user "$(id -u):$(id -g)" "$SARC_BUILD_IMAGE" bash -c \
  'glslc --version; sha256sum /usr/bin/glslc; cmake --version; g++ --version' > "$P/toolchain.txt"
timeout --signal=TERM --kill-after=60 2h "$T/sarc/tools/build.sh" --llama "$T" "$P" > "$P/build.log" 2>&1
python3 - "$P" <<'PY'
import hashlib, pathlib, sys
p = pathlib.Path(sys.argv[1])
binaries = [p / "llama/examples/models/llama/llama_main", p / "tests/test_llama_microbench"]
binaries += sorted(p.rglob("libllama_runner.so"))
for binary in binaries:
    if not binary.is_file():
        raise SystemExit(f"missing binary: {binary}")
with (p / "binaries.sha256").open("w") as out:
    for binary in binaries:
        out.write(f"{hashlib.sha256(binary.read_bytes()).hexdigest()}  {binary}\n")
PY
for b in llama backend; do
  mapfile -t SPVS < <(find "$P/$b" -path '*vulkan_compute_shaders*' -name '*.spv' -printf '%h\n' | sort -u)
  [[ ${#SPVS[@]} == 1 ]] || { echo "expected one SPIR-V directory in $b"; exit 3; }
  python3 "$ET/sarc/tools/spirv_golden.py" "${SPVS[0]}" "$T/sarc/golden/spirv.json" > "$P/golden-$b.txt" 2>&1
done
no_others "after parent build"
gone_check "after parent build"
df -B1 "$A" > "$P/disk-after.txt"
printf 'PARENT_BUILD_COMPLETE %s\n' "$(date -u +%FT%TZ)"
