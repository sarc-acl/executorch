#!/usr/bin/env bash
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
work=$(realpath "${JETSON_CROSS_WORK:-$recipe/../../out/jetson-cross}")
# orin-fused: JETSON_CROSS_GLSLC=<dir with bin/glslc> compiles the shaders with that glslc instead of the image's
# (the glslc of the x86 build image, with which sarc/golden/spirv.json was made); unset = the recipe as it was.
extra=(); [[ -n ${JETSON_CROSS_GLSLC:-} ]] && extra=(-v "$(realpath "$JETSON_CROSS_GLSLC"):/glslc:ro,Z" -e GLSLC=/glslc/bin/glslc)
exec podman run --rm --userns=keep-id \
  -v "$recipe:/recipe:ro,Z" -v "$work:/work:Z" "${extra[@]}" \
  localhost/et-jetson-cross:jp7.2.1 bash /recipe/build.sh "$@"
