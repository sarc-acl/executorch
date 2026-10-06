#!/bin/bash
# build_probe.sh <tree> <build dir of that tree made by sarc/tools/build.sh --llama>: builds
# backends/vulkan/test/sarc_dev/probe (logits_probe) against <build dir>/llama into <build dir>/probe, in the
# build container. CPU only. The tree is the scratch tree the runner was built from (hooks applied or not).
set -euo pipefail
S=$(realpath "$1"); OUT=$(realpath "$2"); ROOT=${SARC_MOUNT_ROOT:-$(dirname "$S")}
podman run --rm --userns=keep-id --security-opt label=disable -v "$ROOT:$ROOT" -e S="$S" -e OUT="$OUT" \
  -e JOBS="${SARC_JOBS:-12}" localhost/et-vk-build:rocky10 bash -euo pipefail -c '
cd "$S"; export PYTHONPATH=$(dirname "$S")
cmake backends/vulkan/test/sarc_dev/probe -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OUT/llama \
  -DCMAKE_FIND_ROOT_PATH=$OUT/llama -DEXECUTORCH_ROOT=$S -B$OUT/probe
cmake --build $OUT/probe -j$JOBS
ls -la $OUT/probe/logits_probe'
