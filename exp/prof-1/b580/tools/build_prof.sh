#!/bin/bash
# build_prof.sh <tag>: container build (sarc/tools/build.sh unchanged: backend + dev tests, no llama) of the working
# copy's HEAD into <box>/.artifacts/build/<tag>. The tree must be clean. Holds the desktop-build lock exclusive
# (never during a measurement). Records commit, image, golden check, binary hash in <tag>.src.txt, and compares
# every SPIR-V file with the campaign's final build (topic7 of b580-fused) in <tag>.spirv-parity.txt.
# After linexp-1's build_exp.sh. A tag is built once.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")"); ET=$(cd $HERE/../../../.. && pwd); BOX=$(dirname $ET); A=$BOX/.artifacts; TAG=${1:?tag}
REF=${REF:-/mnt/linux-share/hmz-campaigns/b580-fused/.artifacts/build/topic7/backend/vulkan_compute_shaders}
mkdir -p $A/build; P=$A/build/$TAG.src.txt; [[ -e $P ]] && { echo "tag $TAG exists" >&2; exit 2; }
[[ -z $(git -C $ET status --porcelain --ignore-submodules=dirty) ]] || { echo "working copy not clean" >&2; exit 2; }
exec 8>>$HOME/.cache/gpu-lab/lock-desktop-build; flock -x -w 14400 8 || { echo "build lock busy" >&2; exit 75; }
unset TMPDIR; IMAGE=${SARC_BUILD_IMAGE:-localhost/et-vk-build:rocky10}
{ echo "tag=$TAG"; echo "commit=$(git -C $ET rev-parse HEAD)"; echo "tree=$(git -C $ET rev-parse HEAD^{tree})"; echo "image=$IMAGE $(podman image inspect --format '{{.Id}}' $IMAGE)"; echo "start=$(date -u +%FT%TZ)"; } > $P
SARC_MOUNT_ROOT=$BOX $ET/sarc/tools/build.sh $ET $A/build/$TAG > $A/build/$TAG.log 2>&1; rc=$?
grep -q SARC_BUILD_OK $A/build/$TAG.log || rc=${rc/#0/1}; echo "build_rc=$rc" >> $P
if [[ $rc == 0 ]]; then
  python3 $ET/sarc/tools/spirv_golden.py $A/build/$TAG/backend/vulkan_compute_shaders $ET/sarc/golden/spirv.json > $A/build/$TAG.golden.txt 2>&1
  echo "golden_rc=$? ($(tail -1 $A/build/$TAG.golden.txt))" >> $P; sha256sum $A/build/$TAG/tests/test_llama_microbench >> $P
  N=$A/build/$TAG/backend/vulkan_compute_shaders
  { same=0; diff=0; onlyref=0; onlynew=0
    for f in $REF/*.spv; do b=$(basename $f); if [[ -e $N/$b ]]; then if cmp -s $f $N/$b; then same=$((same+1)); else diff=$((diff+1)); echo "DIFFERENT $b"; fi; else onlyref=$((onlyref+1)); echo "ONLY_IN_REFERENCE $b"; fi; done
    for f in $N/*.spv; do b=$(basename $f); [[ -e $REF/$b ]] || { onlynew=$((onlynew+1)); echo "ONLY_IN_THIS_BUILD $b $(sha256sum < $f | cut -c1-16)"; }; done
    echo "reference=$REF"; echo "identical=$same different=$diff only_in_reference=$onlyref only_in_this_build=$onlynew"; } > $A/build/$TAG.spirv-parity.txt
  echo "spirv_parity=$(tail -1 $A/build/$TAG.spirv-parity.txt)" >> $P
fi
echo "dirty_after=$(git -C $ET status --porcelain --ignore-submodules=dirty | wc -l)" >> $P; echo "end=$(date -u +%FT%TZ)" >> $P; tail -7 $P; exit $rc
