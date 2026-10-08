#!/usr/bin/env bash
# build-native.sh (copy of the workspace tools/sarc-build-native.sh, adapted: see proposal.md) <executorch dir> <host|android> [--llama] [--etdump] [--inspect] [--no-tests]
#   --no-tests: only llama_main (with --llama/--etdump); skips backends/vulkan/test/sarc_dev (absent in stock 1.5).
#   --etdump: llama_main with ETDump (devtools + event tracer; enables the Vulkan query pool -- for
#   dispatch verification, not for tok/s) into $B/llama-etdump.
# Native (no-container) mirror of sarc/tools/build.sh: runtime+Vulkan install, backends/vulkan/test/sarc_dev
# (test_llama_microbench, test_sarc_select), optional llama_main. NOT golden-faithful: uses the local
# Vulkan SDK glslc, not the pinned container shaderc v2023.8, so SPIR-V differs from sarc/golden.
set -euo pipefail
ET=$1; TGT=$2; shift 2; LLAMA=0; INSPECT=0; ETD=0; NOTESTS=0
for a in "$@"; do case $a in --llama) LLAMA=1;; --etdump) LLAMA=1; ETD=1;; --inspect) INSPECT=1;; --no-tests) NOTESTS=1;; esac; done
cd "$ET"
# Per-tree venv, set up once: uv venv --seed --python 3.12 --managed-python; source .venv/bin/activate;
# ./install_executorch.sh --minimal. It provides torchgen (without it Codegen.cmake globs "/*.py" -- the whole
# filesystem, NFS included) and CMake. No NFS in the build: no NFS dirs on PATH (<toolchain-share>'s cmake reads its
# modules over NFS), ccache dir addressed directly rather than through the ~/.ccache symlink.
VENV=${VENV:-$ET/.venv}; [ -x $VENV/bin/python ] || { echo "no venv at $VENV"; exit 1; }
export VIRTUAL_ENV=$VENV PATH=$VENV/bin:/usr/bin:/bin CCACHE_DIR=<local-home>/.ccache
PY=${PY:-$VENV/bin/python}
$PY -c 'import torchgen, yaml' || { echo "PY=$PY lacks torchgen/pyyaml (run install_executorch.sh --minimal)"; exit 1; }
GL=${GLSLC:-<vulkan-sdk>/glslc}
NDK=${NDK:-<android-ndk>}
A=(); [ "$TGT" = android ] && A=(-DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28)
SUF=""; CXXF="-include algorithm"; [ $INSPECT = 1 ] && { SUF=-inspect; CXXF="$CXXF -DETVK_INSPECT_PIPELINES"; }
B=${OUT_DIR:-$ET/cmake-out-$TGT$SUF}; T=$B/sarc_dev; J=${J:-8}
# ShaderLibrary.cmake's DEPENDS globs glsl/*.yaml|glsl non-recursively, so edits under glsl/sarc*/ never trigger
# codegen; drop the generated spv.cpp so every build regenerates (as sarc/tools/build.sh does).
regen() { find "$1" -path '*vulkan_compute_shaders/spv.cpp' -delete 2>/dev/null || true; }
GATE="$(dirname "$(readlink -f "$0")")/gate_launcher.sh"   # waits while a timed session runs (R5), then ccache
CC=(-DCMAKE_C_COMPILER_LAUNCHER="$GATE;ccache" -DCMAKE_CXX_COMPILER_LAUNCHER="$GATE;ccache"
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5)   # CMake 4 rejects third-party cmake_minimum_required < 3.5
if [ $LLAMA = 1 ]; then
  L=$B/llama; DT=()
  # flatcc (devtools) builds libflatccrt.a INTO the source tree and never rebuilds it for another arch:
  # drop it so host and android ETDump builds of one tree don't link each other's library.
  [ $ETD = 1 ] && { L=$B/llama-etdump; DT=(-DEXECUTORCH_BUILD_DEVTOOLS=ON -DEXECUTORCH_ENABLE_EVENT_TRACER=ON)
    rm -f "$ET/third-party/flatcc/lib/libflatccrt.a"; }
  cmake . -DEXECUTORCH_BUILD_PRESET_FILE=$ET/tools/cmake/preset/llm.cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$L -DEXECUTORCH_BUILD_VULKAN=ON -DGLSLC_PATH=$GL -DPYTHON_EXECUTABLE=$PY \
    -DCMAKE_CXX_FLAGS="$CXXF" "${DT[@]}" "${CC[@]}" "${A[@]}" -B$L
  regen $L; cmake --build $L -j$J --target install
  cmake examples/models/llama -DCMAKE_BUILD_TYPE=Release -DCMAKE_FIND_ROOT_PATH=$L -DCMAKE_PREFIX_PATH=$L \
    -DEXECUTORCH_BUILD_VULKAN=ON -DPYTHON_EXECUTABLE=$PY -DCMAKE_CXX_FLAGS="$CXXF" "${CC[@]}" "${A[@]}" -B$L/examples/models/llama
  cmake --build $L/examples/models/llama -j$J
fi
[ $NOTESTS = 1 ] && { ls -la $L/examples/models/llama/llama_main; echo BUILD_OK; exit 0; }
cmake . -DCMAKE_INSTALL_PREFIX=$B -DEXECUTORCH_BUILD_VULKAN=ON -DPYTHON_EXECUTABLE=$PY -DGLSLC_PATH=$GL \
  -DCMAKE_CXX_FLAGS="$CXXF" "${CC[@]}" "${A[@]}" -B$B
regen $B; cmake --build $B -j$J --target install
cmake backends/vulkan/test/sarc_dev -DCMAKE_PREFIX_PATH=$B -DCMAKE_FIND_ROOT_PATH=$B -DCMAKE_BUILD_TYPE=Debug \
  -DEXECUTORCH_ROOT=$ET -DCMAKE_CXX_FLAGS="$CXXF" "${CC[@]}" "${A[@]}" -B$T
cmake --build $T -j$J
ls -la $T/test_llama_microbench $T/test_sarc_select 2>/dev/null || find $T -maxdepth 2 -type f -perm -u+x -name "test_*"
echo BUILD_OK
