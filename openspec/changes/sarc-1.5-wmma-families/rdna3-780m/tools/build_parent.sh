#!/bin/bash
set -euo pipefail
ROOT=$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)
TOOLS=$ROOT/openspec/changes/sarc-1.5-wmma-families/rdna3-780m/tools
ART=$(dirname "$ROOT")/.artifacts
COMMIT=617897184529ff180af491ee0d5d7de778cbb6c6
TAG=$1
[[ $TAG =~ ^rdna3-780m-parent-[a-zA-Z0-9-]+$ ]] || exit 2
: "${SARC_BUILD_IMAGE:?set the pinned image digest}"
SRC=$ART/exports/$TAG
OUT=$ART/build/$TAG
[[ ! -e $SRC && ! -e $OUT ]] || { echo 'build tag already exists'; exit 2; }
mkdir -p "$ART/exports" "$ART/build" "$ART/tmp"
export TMPDIR=$ART/tmp SARC_MOUNT_ROOT=$(dirname "$ROOT") SARC_JOBS=8
bash "$TOOLS/export_recursive.sh" "$COMMIT" "$SRC" "$ART/submodule-stores"
mkdir "$OUT"
git -C "$ROOT" rev-parse HEAD > "$OUT/ORCHESTRATOR_COMMIT"
cp "$SRC/COMMIT" "$SRC/MANIFEST" "$OUT/"
podman image inspect "$SARC_BUILD_IMAGE" > "$OUT/image.json"
podman run --rm --userns=keep-id --security-opt label=disable \
  "$SARC_BUILD_IMAGE" bash -euo pipefail -c \
  'python3 --version; cmake --version; glslc --version; sha256sum "$(command -v glslc)"' \
  > "$OUT/compiler.txt" 2>&1
sha256sum "$SRC/MANIFEST" "$SRC/executorch/sarc/tools/build.sh" \
  "$SRC/executorch/sarc/golden/spirv.json" > "$OUT/source-hashes.txt"
bash "$SRC/executorch/sarc/tools/build.sh" --llama "$SRC/executorch" "$OUT"
for build in backend llama; do
  python3 "$ROOT/sarc/tools/spirv_golden.py" "$OUT/$build/vulkan_compute_shaders" \
    "$SRC/executorch/sarc/golden/spirv.json" > "$OUT/golden-$build.txt"
done
find "$OUT/llama" "$OUT/tests" -type f \
  \( -name llama_main -o -name libllama_runner.so -o -name test_llama_microbench \) \
  -print0 | sort -z | xargs -0 sha256sum > "$OUT/binaries.sha256"
echo 'PARENT_BUILD_AND_GOLDEN_COMPLETE'
