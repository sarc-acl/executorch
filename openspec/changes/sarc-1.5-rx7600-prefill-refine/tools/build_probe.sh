#!/bin/bash
# build_probe.sh <build tag>: builds backends/vulkan/test/sarc_dev/probe (logits_probe) of the exported tree of
# <build tag> against build/rx7600/<tag>/llama into build/rx7600/<tag>/probe, natively (podman cannot run here).
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$A/src/rx7600/$1/executorch; OUT=$A/build/rx7600/$1; export PYTHONPATH=$(dirname "$S") CCACHE_DIR=$A/ccache
cd "$S"
cmake backends/vulkan/test/sarc_dev/probe -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OUT/llama \
  -DCMAKE_FIND_ROOT_PATH=$OUT/llama -DEXECUTORCH_ROOT=$S -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B$OUT/probe
cmake --build $OUT/probe -j${SARC_JOBS:-24}
ls -la $OUT/probe/logits_probe
