#!/bin/bash
# build-hook.sh <tag> <patch file> [commit]: build-both.sh for a commit with a LOCAL, UNCOMMITTED patch applied to
# its export (never to the working copy). For measuring a kernel that needs a release-zone hook the campaign may
# not commit (CAMPAIGN.md: "write down the smallest hook that would be needed and measure through a local patch
# that you do NOT commit"). build/<tag>.src.txt records `hook_patch=<file> sha256=...`; a session staged from
# such a build is a hook measurement, never a candidate.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"
TAG=${1:?tag}; PATCH=$(readlink -f "${2:?patch file}"); REV=${3:-HEAD}; [[ -f $PATCH ]] || { echo "no patch $PATCH" >&2; exit 2; }
IMAGE=${SARC_BUILD_IMAGE:-localhost/et-vk-build:rocky10}
SHA=$(git -C $ET rev-parse --verify "$REV^{commit}") || { echo "unknown commit $REV" >&2; exit 2; }
mkdir -p $A/build $A/src || { echo "cannot create $A/build" >&2; exit 2; }
P=$A/build/$TAG.src.txt; SRC=$A/src/$TAG/executorch
[[ -e $P || -e $SRC ]] && { echo "tag $TAG already exists ($P); use a new tag" >&2; exit 2; }
podman image exists $IMAGE || { echo "missing image $IMAGE: podman build -t $IMAGE -f $TOOLS/Containerfile $TOOLS" >&2; exit 2; }
export XE2_EXPORT_MANIFEST=$A/src/$TAG.export-manifest
export_commit $ET $SHA $SRC || { echo "export failed" >&2; exit 2; }
(cd $SRC && git apply -p1 --verbose $PATCH) > $A/build/$TAG.patch.log 2>&1   # the export is not a git tree; git apply works on plain files || { echo "patch failed, see $A/build/$TAG.patch.log" >&2; exit 2; }
{ echo "tag=$TAG"; echo "hook_patch=$PATCH sha256=$(sha256sum < $PATCH | cut -c1-64) (LOCAL, UNCOMMITTED: hook measurement, not a candidate build)"; echo "commit=$SHA"; echo "tree=$(git -C $ET rev-parse $SHA^{tree})"; echo "subject=$(git -C $ET log -1 --format=%s $SHA)"
  echo "requested=$REV parent_commit=$PARENT_COMMIT"; echo "export_manifest=$(wc -l < $XE2_EXPORT_MANIFEST) trees sha256=$(sha256sum < $XE2_EXPORT_MANIFEST | cut -c1-64)"
  echo "image=$IMAGE $(podman image inspect --format '{{.Id}}' $IMAGE)"; echo "glslc=$(podman run --rm $IMAGE glslc --version | tr '\n' ' ')"
  echo "start=$(date -u +%FT%TZ)"; } > $P 2>&1
ok=1
$ET/sarc/tools/build.sh --llama $SRC $A/build/$TAG > $A/build/$TAG.log 2>&1; rc=$?
grep -q SARC_BUILD_OK $A/build/$TAG.log || rc=${rc/#0/1}; echo "main_rc=$rc" >> $P; [[ $rc == 0 ]] || ok=0
$ET/sarc/tools/build.sh --llama --traced --no-tests $SRC $A/build/$TAG-traced > $A/build/$TAG-traced.log 2>&1; rc=$?
grep -q SARC_BUILD_OK $A/build/$TAG-traced.log || rc=${rc/#0/1}; echo "traced_rc=$rc" >> $P; [[ $rc == 0 ]] || ok=0
if [[ $ok == 1 ]]; then
  python3 $ET/sarc/tools/spirv_golden.py $A/build/$TAG/backend/vulkan_compute_shaders $ET/sarc/golden/spirv.json > $A/build/$TAG.golden.txt 2>&1
  rc=$?; echo "golden_rc=$rc ($(tail -1 $A/build/$TAG.golden.txt))" >> $P; [[ $rc == 0 ]] || ok=0
  sha256sum $A/build/$TAG/llama/examples/models/llama/llama_main $A/build/$TAG/tests/test_llama_microbench \
    $A/build/$TAG-traced/llama/examples/models/llama/llama_main >> $P 2>&1 || ok=0
fi
echo "end=$(date -u +%FT%TZ)" >> $P
if [[ $ok == 1 ]]; then echo BUILD_BOTH_OK >> $P; else echo BUILD_BOTH_FAILED >> $P; fi
tail -8 $P; [[ $ok == 1 ]]
