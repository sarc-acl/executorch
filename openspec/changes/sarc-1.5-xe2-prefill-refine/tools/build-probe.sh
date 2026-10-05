#!/bin/bash
# build-probe.sh <build tag>: build tools/logits_probe against the llama install tree of build/<tag>, in the
# build container, into build/<tag>/probe/logits_probe. The source tree is the build's own export
# (src/<tag>/executorch) with the campaign's tools/logits_probe sources copied beside it, so a parent build gets
# the same probe program as a topic build. No GPU is used; do not run it during a measurement.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; TAG=${1:?build tag}; IMAGE=${SARC_BUILD_IMAGE:-localhost/et-vk-build:rocky10}
B=$A/build/$TAG; SRC=$A/src/$TAG/executorch; P=$A/src/$TAG/logits_probe
grep -qx BUILD_BOTH_OK $A/build/$TAG.src.txt || { echo "build/$TAG is not a successful build" >&2; exit 2; }
rm -rf $P $B/probe; mkdir -p $P && cp $TOOLS/logits_probe/main.cpp $TOOLS/logits_probe/CMakeLists.txt $P/ || exit 2
podman run --rm --userns=keep-id --security-opt label=disable -v "$XE2_ROOT:$XE2_ROOT" "$IMAGE" bash -euo pipefail -c "
  cmake $P -B$B/probe -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_ROOT=$SRC -DCMAKE_PREFIX_PATH=$B/llama -DCMAKE_FIND_ROOT_PATH=$B/llama
  cmake --build $B/probe -j12" > $A/build/$TAG.probe.log 2>&1; rc=$?
[[ $rc == 0 && -x $B/probe/logits_probe ]] && { sha256sum $B/probe/logits_probe $P/main.cpp | tee $A/build/$TAG.probe.txt; echo PROBE_BUILD_OK; exit 0; }
tail -20 $A/build/$TAG.probe.log; echo PROBE_BUILD_FAILED; exit 1
