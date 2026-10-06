#!/bin/bash
# build-native.sh [--traced] [--no-tests] <exported tree>/executorch <out>: native mirror of sarc/tools/build.sh
# --llama for this host (podman cannot run here: /etc/subgid maps the user GID). Same cmake invocations and
# layout (<out>/{backend,tests,llama}); host gcc, /tool/pkg python 3.12 (torch, yaml), Vulkan SDK 1.4.350.1 glslc
# instead of the pinned container's shaderc v2023.8: shipped SPIR-V does not match sarc/golden (golden: pending).
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
TRACED=0; TESTS=1
while [[ $1 == --* ]]; do case $1 in --traced) TRACED=1 ;; --no-tests) TESTS=0 ;; *) exit 2 ;; esac; shift; done
SRC=$(realpath "$1"); OUT=$(realpath -m "$2"); mkdir -p "$OUT"; cd "$SRC"
[[ $(basename "$SRC") == executorch ]] || { echo "tree must be named executorch" >&2; exit 2; }
export PYTHONPATH=$(dirname "$SRC") CCACHE_DIR=$A/ccache
PY=/tool/pkg/Python-3.12.9-1/bin/python3; GL=/local/yanwen.xu/vulkan-sdk/1.4.350.1/x86_64/bin/glslc; JOBS=${SARC_JOBS:-24}
CC=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
EXTRA=(); [[ $TRACED == 1 ]] && EXTRA+=(-DEXECUTORCH_BUILD_DEVTOOLS=ON -DEXECUTORCH_ENABLE_EVENT_TRACER=ON)
{ echo "src $(cat $(dirname "$SRC")/COMMIT) traced=$TRACED tests=$TESTS $(date -u +%FT%TZ)"; gcc --version | sed -n 1p
  $GL --version | sed -n 1p; $PY --version; cmake --version | sed -n 1p; } > "$OUT/TOOLCHAIN"
set -x
L=$OUT/llama
cmake . -DEXECUTORCH_BUILD_PRESET_FILE=$SRC/tools/cmake/preset/llm.cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$L -DEXECUTORCH_BUILD_VULKAN=ON -DGLSLC_PATH=$GL -DPYTHON_EXECUTABLE=$PY "${CC[@]}" "${EXTRA[@]}" -B$L
cmake --build $L -j$JOBS --target install
cmake examples/models/llama -DCMAKE_BUILD_TYPE=Release -DCMAKE_FIND_ROOT_PATH=$L -DCMAKE_PREFIX_PATH=$L \
  -DEXECUTORCH_BUILD_VULKAN=ON -DPYTHON_EXECUTABLE=$PY "${CC[@]}" "${EXTRA[@]}" -B$L/examples/models/llama
cmake --build $L/examples/models/llama -j$JOBS
ls -la $L/examples/models/llama/llama_main
if [[ $TESTS == 1 ]]; then
  B=$OUT/backend; T=$OUT/tests
  cmake . -DCMAKE_INSTALL_PREFIX=$B -DEXECUTORCH_BUILD_VULKAN=ON -DPYTHON_EXECUTABLE=$PY "${CC[@]}" -DGLSLC_PATH=$GL -B$B
  cmake --build $B -j$JOBS --target install
  cmake backends/vulkan/test/sarc_dev -DCMAKE_PREFIX_PATH=$B -DCMAKE_FIND_ROOT_PATH=$B \
    -DCMAKE_BUILD_TYPE=Debug -DEXECUTORCH_ROOT=$SRC "${CC[@]}" -B$T
  cmake --build $T -j$JOBS
fi
echo SARC_BUILD_OK
