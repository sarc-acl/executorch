#!/bin/bash
set -euo pipefail
ROOT=$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)
REL=openspec/changes/sarc-1.5-wmma-families/rdna3-780m/tools
ART=$(dirname "$ROOT")/.artifacts
PARENT=$1
TAG=$2
[[ $PARENT =~ ^rdna3-780m-parent-[a-zA-Z0-9-]+$ && $TAG =~ ^rdna3-780m-metadata-[a-zA-Z0-9-]+$ ]] || exit 2
: "${SARC_BUILD_IMAGE:?set the pinned image digest}"
OUT=$ART/helpers/$TAG
[[ ! -e $OUT ]] || { echo 'helper tag already exists'; exit 2; }
mkdir -p "$OUT/source"
COMMIT=$(git -C "$ROOT" rev-parse HEAD)
echo "$COMMIT" > "$OUT/HELPER_COMMIT"
cp "$ART/build/$PARENT/COMMIT" "$OUT/RUNTIME_COMMIT"
cp "$ART/build/$PARENT/MANIFEST" "$OUT/RUNTIME_MANIFEST"
git -C "$ROOT" archive "$COMMIT" "$REL" | tar -x -C "$OUT/source"
TOOLS=$OUT/source/$REL
podman run --rm --userns=keep-id --security-opt label=disable \
  -v "$(dirname "$ROOT"):$(dirname "$ROOT")" \
  -e HELPER_SRC="$TOOLS/metadata" -e HELPER_OUT="$OUT/build" \
  -e PREFIX="$ART/build/$PARENT/llama" "$SARC_BUILD_IMAGE" \
  bash -euxo pipefail -c '
cmake -S "$HELPER_SRC" -B "$HELPER_OUT" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_FIND_ROOT_PATH="$PREFIX"
cmake --build "$HELPER_OUT" -j2
readelf -d "$HELPER_OUT/wmma_metadata" > "$HELPER_OUT/dynamic.txt"
readelf -Ws "$HELPER_OUT/wmma_metadata" > "$HELPER_OUT/symbols.txt"
! grep -Ei "libvulkan|amdgpu|llama_runner" "$HELPER_OUT/dynamic.txt"
! grep -E "VulkanBackend|vkcompute|vkCreate|vkQueue" "$HELPER_OUT/symbols.txt"
'
sha256sum "$OUT/build/wmma_metadata" "$OUT/HELPER_COMMIT" \
  "$OUT/RUNTIME_COMMIT" "$OUT/RUNTIME_MANIFEST" > "$OUT/provenance.sha256"
python3 "$TOOLS/validate_metadata.py" "$OUT/build/wmma_metadata" \
  "$ART/preflight/models.json" "$ART/preflight/model-hashes.txt" "$OUT/runtime-metadata.json"
echo 'CPU_METADATA_VALIDATION_COMPLETE'
