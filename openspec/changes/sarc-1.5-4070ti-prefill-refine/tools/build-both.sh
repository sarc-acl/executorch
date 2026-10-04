#!/bin/bash
# build-both.sh <tag> <commit> [local patch]: build an immutable source tree, never the working copy.
#   1. mktree.sh archives <commit> (and its pinned submodules) into src/<tag>/executorch;
#   2. an optional local patch (a release-zone hook that is measured but never committed, e.g.
#      tools/local-hook-nvidia-sdpa.patch) is applied to that tree only and recorded;
#   3. sarc/tools/build.sh (unmodified, from the tree being built) builds build/<tag> (backend + tests +
#      llama_main) and build/<tag>-traced (ETDump llama_main); the tree is read-only from then on.
# The parent is `build-both.sh parent 6a7cc8cc6`. Provenance (commit, patch and tree hashes, toolchain, binary
# hashes, shipped-SPIR-V comparison) goes to build/<tag>.src.txt. build.sh calls podman: bin/podman is the
# docker shim (podman-shim.sh). Exit status is non-zero when any step fails.
set -uo pipefail
source "$(dirname "$0")/common.sh"; TAG=$1; C=$(git -C $ET rev-parse --verify "$2^{commit}") || exit 2; PATCH=${3:-}
[[ -e $A/build/$TAG.src.txt ]] && { echo "build tag $TAG exists; tags are immutable" >&2; exit 2; }
mkdir -p $A/build $A/bin $A/src; install -m 755 $TOOLS/podman-shim.sh $A/bin/podman
export PATH=$A/bin:$PATH DOCKER_CONFIG=$A/docker-config; mkdir -p $DOCKER_CONFIG
T=$A/src/$TAG/executorch; P=$A/build/$TAG.src.txt
$TOOLS/mktree.sh $C $T > $A/build/$TAG.mktree.log 2>&1 || { echo "mktree failed, see $A/build/$TAG.mktree.log" >&2; exit 3; }
{ echo "tag $TAG built $(date -u +%FT%TZ) on $(hostname)"; echo "commit $C"; echo "source $T (git archive + pinned submodules, $(wc -l < $A/src/$TAG/SUBMODULES) submodules)"; } > $P
if [[ -n $PATCH ]]; then
  need "$PATCH"; chmod -R u+w $T
  patch -d $T -p1 --no-backup-if-mismatch < "$PATCH" >> $A/build/$TAG.mktree.log 2>&1 || { echo "local patch failed" >&2; exit 3; }
  echo "local-patch $TAG: $(basename "$PATCH") sha256 $(sha256sum < "$PATCH" | cut -d' ' -f1) (NOT committed; files: $(grep '^+++ ' "$PATCH" | sed 's/^+++ b\///' | tr '\n' ' '))" >> $P
fi
chmod -R a-w $T; find $T -type d -exec chmod u+w {} +   # cmake writes nothing into the tree, but python may want __pycache__
echo "tree-sha256 $TAG: $(cd $T && find . -type f -not -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)" >> $P
echo "image $(docker image inspect --format '{{.Id}}' localhost/et-vk-build:rocky10 2>&1)" >> $P
rc=0
$T/sarc/tools/build.sh --llama $T $A/build/$TAG > $A/build/$TAG.log 2>&1 || rc=$?; echo "rc=$rc main" >> $P
if [[ $rc == 0 ]]; then
  $T/sarc/tools/build.sh --llama --traced --no-tests $T $A/build/$TAG-traced > $A/build/$TAG-traced.log 2>&1 || rc=$?; echo "rc=$rc traced" >> $P
fi
if [[ $rc == 0 ]]; then
  sha256sum $A/build/$TAG/llama/examples/models/llama/llama_main $A/build/$TAG-traced/llama/examples/models/llama/llama_main \
    $(find $A/build/$TAG/llama $A/build/$TAG-traced/llama -name libllama_runner.so) $A/build/$TAG/tests/test_llama_microbench >> $P || rc=4
  # Shipped variants must be unchanged (other devices' SPIR-V): golden comparison on this build's shaders.
  SPV=$(dirname "$(find $A/build/$TAG/backend -path '*vulkan_compute_shaders*' -name '*.spv' | head -1)")
  python3 $ET/sarc/tools/spirv_golden.py "$SPV" $ET/sarc/golden/spirv.json > $A/build/$TAG.golden.txt 2>&1; g=$?
  echo "spirv_golden rc=$g: $(tail -1 $A/build/$TAG.golden.txt)" >> $P; [[ $g == 0 ]] || rc=5
fi
date -u >> $P; echo "BUILD_BOTH_DONE rc=$rc" >> $P; tail -4 $P; exit $rc
