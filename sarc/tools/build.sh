#!/bin/bash
# Build the Vulkan backend, the SARC dev test binaries and (optionally)
# llama_main on the host, with the pinned glslc (shaderc v2026.2) that the
# SPIR-V golden is recorded with.
#
# usage: sarc/tools/build.sh [--android] [--llama] [--traced] [--no-tests] <executorch-tree> <out-dir>
#   <executorch-tree>  a dev/1.5 checkout or a release export (make-release.sh --export)
#   <out-dir>          build root; host: <out>/{backend,tests,llama}; android: <out>/android-*
#
# Environment: SARC_GLSLC (default /sarc-c/gpusw/users/yanwen.xu/toolchains/shaderc-v2026.2/glslc;
#   build it from shaderc tag v2026.2 with utils/git-sync-deps, see sarc/README.md),
#   SARC_PYTHON (default python3; needs torchgen and pyyaml), ANDROID_NDK
#   (default ~/android-ndk-r30), SARC_JOBS (default 12).
#
# Shaders are regenerated on every run: the ET shader-lib DEPENDS glob is not
# recursive, so edits under glsl/sarc*/ would otherwise not trigger a rebuild.
set -euo pipefail

ANDROID=0; LLAMA=0; TRACED=0; TESTS=1
while [[ $# -gt 0 && $1 == --* ]]; do
  case $1 in
    --android) ANDROID=1 ;;
    --llama) LLAMA=1 ;;
    --traced) TRACED=1 ;;
    --no-tests) TESTS=0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[[ $# -eq 2 ]] || { sed -n '2,16p' "$0"; exit 2; }
ET=$(realpath "$1"); OUT=$(realpath -m "$2")
GL=${SARC_GLSLC:-/sarc-c/gpusw/users/yanwen.xu/toolchains/shaderc-v2026.2/glslc}
PY=${SARC_PYTHON:-python3}
NDK=${ANDROID_NDK:-$HOME/android-ndk-r30}
JOBS=${SARC_JOBS:-12}
GLSLC_VERSION="shaderc v2026.2"
[[ $("$GL" --version | head -1) == "$GLSLC_VERSION "* ]] || {
  echo "$GL is not $GLSLC_VERSION (the golden's glslc); set SARC_GLSLC" >&2; exit 2; }
"$PY" -c 'import torchgen, yaml' || { echo "$PY lacks torchgen/pyyaml; set SARC_PYTHON" >&2; exit 2; }
[[ $(basename "$ET") == executorch ]] || { echo "the tree directory must be named executorch" >&2; exit 2; }
mkdir -p "$OUT"
set -x

cd "$ET"; export PYTHONPATH=$(dirname "$ET")
EXTRA=()
[[ $TRACED == 1 ]] && EXTRA+=(-DEXECUTORCH_BUILD_DEVTOOLS=ON -DEXECUTORCH_ENABLE_EVENT_TRACER=ON)
if [[ $ANDROID == 1 ]]; then
  A=(-DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28)
  B=$OUT/android-backend; T=$OUT/android-tests
else
  A=(); B=$OUT/backend; T=$OUT/tests
fi
regen() { find "$1" -path "*vulkan_compute_shaders*" -name "spv.cpp" -delete 2>/dev/null || true; }
if [[ $LLAMA == 1 && $ANDROID == 0 ]]; then
  L=$OUT/llama
  regen "$L"
  cmake . -DEXECUTORCH_BUILD_PRESET_FILE=$ET/tools/cmake/preset/llm.cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$L -DEXECUTORCH_BUILD_VULKAN=ON -DGLSLC_PATH=$GL \
    -DPYTHON_EXECUTABLE=$PY "${EXTRA[@]}" -B$L
  cmake --build $L -j$JOBS --target install
  cmake examples/models/llama -DCMAKE_BUILD_TYPE=Release -DCMAKE_FIND_ROOT_PATH=$L -DCMAKE_PREFIX_PATH=$L \
    -DEXECUTORCH_BUILD_VULKAN=ON -DPYTHON_EXECUTABLE=$PY "${EXTRA[@]}" -B$L/examples/models/llama
  cmake --build $L/examples/models/llama -j$JOBS
  ls -la $L/examples/models/llama/llama_main
fi
if [[ $TESTS == 1 ]]; then
  regen "$B"
  cmake . -DCMAKE_INSTALL_PREFIX=$B -DEXECUTORCH_BUILD_VULKAN=ON -DPYTHON_EXECUTABLE=$PY \
    "${A[@]}" -DGLSLC_PATH=$GL -B$B
  cmake --build $B -j$JOBS --target install
  if [[ -d backends/vulkan/test/sarc_dev ]]; then
    cmake backends/vulkan/test/sarc_dev -DCMAKE_PREFIX_PATH=$B -DCMAKE_FIND_ROOT_PATH=$B \
      -DCMAKE_BUILD_TYPE=Debug -DEXECUTORCH_ROOT=$ET "${A[@]}" -B$T
    cmake --build $T -j$JOBS
  fi
fi
echo SARC_BUILD_OK
