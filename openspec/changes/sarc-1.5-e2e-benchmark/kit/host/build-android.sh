#!/usr/bin/env bash
# build-android.sh: cross-compile the e2e binaries of one ExecuTorch tree for an Android phone (arm64-v8a)
# with the NDK, natively (no container). Used for the S26 (Adreno 840) contribution; any Android GPU works.
#
# usage: build-android.sh --tree <executorch dir> [--ndk DIR] [--glslc PATH] [--out DIR] [-j N]
#                         [--llama] [--etdump] [--probe] [--tests] [--clean-path]
#   --llama   runtime + Vulkan install into <out>/llama, then llama_main
#             -> <out>/llama/examples/models/llama/llama_main
#   --etdump  as --llama with ETDump (devtools + event tracer; enables the Vulkan query pool, so use it for
#             M2 traces, not for tok/s) into <out>/llama-etdump
#   --probe   kit/logits_probe linked against <out>/llama (needs --llama now or earlier) -> <out>/probe/logits_probe
#   --tests   runtime install into <out> and backends/vulkan/test/sarc_dev (test_llama_microbench,
#             test_sarc_select) -> <out>/sarc_dev/ (SARC trees only; stock 1.5 has no sarc_dev)
#   --out     build root (default <tree>/cmake-out-android)
#   --ndk     NDK r26+ (default $ANDROID_NDK, then $ANDROID_NDK_HOME)
#   --glslc   glslc for the Vulkan shaders (default $GLSLC, then the first glslc on PATH)
#   --clean-path  PATH=<tree>/.venv/bin:/usr/bin:/bin for the build (keeps slow network mounts off PATH)
# Needs a per-tree venv with torchgen and pyyaml (uv venv --seed --python 3.12; ./install_executorch.sh
# --minimal), or $PYTHON pointing at such a python. Uses ccache when it is installed.
# Not golden-faithful: SPIR-V from a local glslc differs from sarc/tools/build.sh's pinned shaderc.
# Push to the phone: llama_main alone (static); the probe alone; test_llama_microbench alone.
set -euo pipefail
ET=""; NDK=${ANDROID_NDK:-${ANDROID_NDK_HOME:-}}; GL=${GLSLC:-}; B=""; J=${J:-$(nproc)}
LLAMA=0; ETD=0; PROBE=0; TESTS=0; CLEANPATH=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --tree) ET=$2; shift ;; --ndk) NDK=$2; shift ;; --glslc) GL=$2; shift ;; --out) B=$2; shift ;;
    -j) J=$2; shift ;; --llama) LLAMA=1 ;; --etdump) ETD=1 ;; --probe) PROBE=1 ;; --tests) TESTS=1 ;;
    --clean-path) CLEANPATH=1 ;; -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
[[ -n $ET ]] || { sed -n '2,21p' "$0"; exit 2; }
ET=$(cd "$ET" && pwd); KIT=$(cd "$(dirname "$0")/.." && pwd)
[[ -n $NDK && -f $NDK/build/cmake/android.toolchain.cmake ]] || { echo "no NDK (--ndk or ANDROID_NDK)" >&2; exit 1; }
[[ -n $GL ]] || GL=$(command -v glslc || true); [[ -x $GL ]] || { echo "no glslc (--glslc or GLSLC)" >&2; exit 1; }
PY=${PYTHON:-$ET/.venv/bin/python}; [[ -x $PY ]] || { echo "no python at $PY (set up <tree>/.venv or PYTHON)" >&2; exit 1; }
[[ $CLEANPATH == 1 ]] && export PATH=$(dirname "$PY"):/usr/bin:/bin
export VIRTUAL_ENV=$(dirname "$(dirname "$PY")")
"$PY" -c 'import torchgen, yaml' || { echo "$PY lacks torchgen/pyyaml (run install_executorch.sh --minimal)" >&2; exit 1; }
B=${B:-$ET/cmake-out-android}; mkdir -p "$B"; B=$(cd "$B" && pwd)
A=(-DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28)
CXXF="-include algorithm"
CC=(-DCMAKE_POLICY_VERSION_MINIMUM=3.5)   # CMake 4 rejects third-party cmake_minimum_required < 3.5
command -v ccache > /dev/null && CC+=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
cd "$ET"
# ShaderLibrary.cmake's DEPENDS globs glsl/*.yaml|glsl non-recursively, so edits under glsl/sarc*/ never
# trigger codegen; drop the generated spv.cpp so every build regenerates (as sarc/tools/build.sh does).
regen() { find "$1" -path '*vulkan_compute_shaders/spv.cpp' -delete 2>/dev/null || true; }
llama() {  # llama <install prefix> [extra cmake args]
  local L=$1; shift
  cmake . -DEXECUTORCH_BUILD_PRESET_FILE=$ET/tools/cmake/preset/llm.cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$L -DEXECUTORCH_BUILD_VULKAN=ON -DGLSLC_PATH=$GL -DPYTHON_EXECUTABLE=$PY \
    -DCMAKE_CXX_FLAGS="$CXXF" "$@" "${CC[@]}" "${A[@]}" -B$L
  regen $L; cmake --build $L -j$J --target install
  cmake examples/models/llama -DCMAKE_BUILD_TYPE=Release -DCMAKE_FIND_ROOT_PATH=$L -DCMAKE_PREFIX_PATH=$L \
    -DEXECUTORCH_BUILD_VULKAN=ON -DPYTHON_EXECUTABLE=$PY -DCMAKE_CXX_FLAGS="$CXXF" "${CC[@]}" "${A[@]}" \
    -B$L/examples/models/llama
  cmake --build $L/examples/models/llama -j$J
  ls -la $L/examples/models/llama/llama_main
}
[[ $LLAMA == 1 ]] && llama $B/llama
if [[ $ETD == 1 ]]; then
  # flatcc (devtools) builds libflatccrt.a INTO the source tree and never rebuilds it for another arch:
  # drop it so host and Android ETDump builds of one tree do not link each other's library.
  rm -f "$ET/third-party/flatcc/lib/libflatccrt.a"
  llama $B/llama-etdump -DEXECUTORCH_BUILD_DEVTOOLS=ON -DEXECUTORCH_ENABLE_EVENT_TRACER=ON
fi
if [[ $PROBE == 1 ]]; then
  [[ -d $B/llama/lib/cmake/ExecuTorch ]] || { echo "no install prefix $B/llama (add --llama)" >&2; exit 1; }
  cmake "$KIT/logits_probe" -B$B/probe -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_ROOT="$ET" \
    -DCMAKE_PREFIX_PATH=$B/llama -DCMAKE_FIND_ROOT_PATH=$B/llama -DCMAKE_CXX_FLAGS="$CXXF" "${CC[@]}" "${A[@]}"
  cmake --build $B/probe -j$J
  ls -la $B/probe/logits_probe
fi
if [[ $TESTS == 1 ]]; then
  cmake . -DCMAKE_INSTALL_PREFIX=$B -DEXECUTORCH_BUILD_VULKAN=ON -DPYTHON_EXECUTABLE=$PY -DGLSLC_PATH=$GL \
    -DCMAKE_CXX_FLAGS="$CXXF" "${CC[@]}" "${A[@]}" -B$B
  regen $B; cmake --build $B -j$J --target install
  cmake backends/vulkan/test/sarc_dev -DCMAKE_PREFIX_PATH=$B -DCMAKE_FIND_ROOT_PATH=$B -DCMAKE_BUILD_TYPE=Debug \
    -DEXECUTORCH_ROOT=$ET -DCMAKE_CXX_FLAGS="$CXXF" "${CC[@]}" "${A[@]}" -B$B/sarc_dev
  cmake --build $B/sarc_dev -j$J
  ls -la $B/sarc_dev/test_llama_microbench
fi
echo BUILD_OK
