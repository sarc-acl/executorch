#!/bin/bash
# build-orin.sh <tag> <commit> [local patch]: cross-build an immutable source tree for the Orin, never the
# working copy. Workstation side.
#   1. mktree.sh archives <commit> (and its pinned submodules) into build/<tag>/source/executorch;
#   2. an optional local patch (a release-zone hook that is measured but never committed) is applied to that
#      tree only and recorded;
#   3. tools/jetson-cross/container.sh (the recipe of the campaign that produced the Orin numbers of cells.csv,
#      image localhost/et-jetson-cross:jp7.2.1, 8 jobs) builds build/<tag>/bundle/{llama_main,
#      libllama_runner.so,test_llama_microbench,vulkan-smoke}. One build serves timed and traced runs: the
#      runner links ETDump (as in that campaign) and traces only when --etdump_path is given.
# BASE_TAG=<tag> makes the tree with mktree_from.sh (hard links to that tag's tree + the changed files).
# The whole build runs under an exclusive flock on ~/.cache/gpu-lab/lock-desktop-build (the workstation's GPU
# is measured by another campaign). The parent is `build-orin.sh parent 8973ced76`.
# Provenance goes to build/<tag>.src.txt: commit, patch and tree hashes, image id, binary hashes, the
# spirv_golden.py result (informative: the cross image's glslc is not the one the goldens were made with) and
# the sha256 of every compiled shader (build/<tag>.spv.sha256), which shipped.sh compares between builds.
set -uo pipefail
source "$(dirname "$0")/common.sh"; TAG=$1; C=$(git -C $ET rev-parse --verify "$2^{commit}") || exit 2; PATCH=${3:-}
[[ $SIDE == ws ]] || { echo "workstation only" >&2; exit 2; }
[[ -e $A/build/$TAG.src.txt ]] && { echo "build tag $TAG exists; tags are immutable" >&2; exit 2; }
W=$A/build/$TAG; T=$W/source/executorch; P=$A/build/$TAG.src.txt; mkdir -p $W
if [[ -n ${BASE_TAG:-} ]]; then MK="$TOOLS/mktree_from.sh $C $T $BASE_TAG"; else MK="$TOOLS/mktree.sh $C $T"; fi
# A tree that mktree completed earlier for the same commit (its COMMIT file is written last) is used as it is.
if [[ -f $W/source/COMMIT && $(cat $W/source/COMMIT) == "$C" && ! -e $W/build ]]; then echo "tree $C -> $T (made by an earlier, interrupted invocation; reused)" >> $A/build/$TAG.mktree.log
else $MK > $A/build/$TAG.mktree.log 2>&1 || { echo "mktree failed, see $A/build/$TAG.mktree.log" >&2; exit 3; }; fi
{ echo "tag $TAG built $(date -u +%FT%TZ) on $(hostname) for aarch64 (Jetson Orin, L4T R39.2.1)"; echo "commit $C"
  echo "source $T ($(tail -1 $A/build/$TAG.mktree.log | cut -c1-200); $(wc -l < $W/source/SUBMODULES) submodules)"; } > $P
if [[ -n $PATCH ]]; then
  need "$PATCH"; chmod -R u+w $T
  patch -d $T -p1 --no-backup-if-mismatch < "$PATCH" >> $A/build/$TAG.mktree.log 2>&1 || { echo "local patch failed" >&2; exit 3; }
  echo "local-patch $TAG: $(basename "$PATCH") sha256 $(sha256sum < "$PATCH" | cut -d' ' -f1) (NOT committed; files: $(grep '^+++ ' "$PATCH" | sed 's/^+++ b\///' | tr '\n' ' '))" >> $P
fi
chmod -R a-w $T; find $T -type d -exec chmod u+w {} +   # python may want __pycache__
# Hashing the whole tree takes over ten minutes on this NAS. A tree made by mktree_from.sh is the base tree (hashed
# once) plus changed files that mktree_from.sh verified by git blob hash; its hash is not recomputed.
if [[ -n ${BASE_TAG:-} && -z $PATCH ]]; then echo "tree-sha256 $TAG: not recomputed: hard links to $BASE_TAG ($(grep -m1 '^tree-sha256' $A/build/$BASE_TAG.src.txt | cut -d' ' -f3 | cut -c1-16)...) + changed files verified by blob hash" >> $P
else echo "tree-sha256 $TAG: $(cd $T && find . -type f -not -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)" >> $P; fi
echo "image $(podman image inspect --format '{{.Id}}' localhost/et-jetson-cross:jp7.2.1 2>&1)" >> $P
echo "recipe $(cd $TOOLS/jetson-cross && sha256sum build.sh container.sh aarch64.cmake | sha256sum | cut -d' ' -f1)" >> $P
rc=0; hold_wait
flock ~/.cache/gpu-lab/lock-desktop-build env JETSON_CROSS_WORK=$W $TOOLS/jetson-cross/container.sh > $A/build/$TAG.log 2>&1 || rc=$?
echo "rc=$rc cross build" >> $P
if [[ $rc == 0 ]]; then
  sha256sum $W/bundle/* >> $P || rc=4
  SPV=$(dirname "$(find $W/build -path '*vulkan_compute_shaders*' -name '*.spv' | head -1)")
  ( cd "$SPV" && sha256sum *.spv ) > $A/build/$TAG.spv.sha256
  python3 $ET/sarc/tools/spirv_golden.py "$SPV" $ET/sarc/golden/spirv.json > $A/build/$TAG.golden.txt 2>&1; g=$?
  echo "spirv_golden rc=$g: $(tail -1 $A/build/$TAG.golden.txt) ($(grep -c '^DIFF' $A/build/$TAG.golden.txt) DIFF lines; cross glslc $(grep -h -m1 -o 'shaderc[^"]*' $A/build/$TAG.log | head -1))" >> $P
fi
date -u >> $P; echo "BUILD_ORIN_DONE rc=$rc" >> $P; tail -4 $P; exit $rc
