#!/bin/bash
# build_probe.sh <tag>: builds backends/vulkan/test/sarc_dev/probe (logits_probe) of the exported tree of build tag
# <tag> against that tag's llama install tree, cross-compiled like build-native.sh, into build/m51/<tag>/probe.
# Adapted from the 780M's tools/build_probe.sh (native NDK build instead of the container). CPU only.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/dev.sh"
SRC=$ART/src/m51/$1/executorch; B=$ART/build/m51/$1; V=$ART/venv/m51
NDK=${NDK:-<android-ndk>}
export PATH=$V/bin:/usr/bin:/bin CCACHE_DIR=<local-home>/.ccache
while other_timed_session; do sleep 60; done
nice -n 19 cmake "$SRC/backends/vulkan/test/sarc_dev/probe" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$B/llama \
  -DCMAKE_FIND_ROOT_PATH=$B/llama -DEXECUTORCH_ROOT=$SRC -DCMAKE_CXX_FLAGS="-include algorithm" \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B$B/probe
nice -n 19 cmake --build $B/probe -j${J:-8}
ls -la $B/probe/logits_probe
