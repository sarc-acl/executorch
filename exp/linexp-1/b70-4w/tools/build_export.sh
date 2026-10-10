#!/bin/bash
# build_export.sh <tag>: like build_exp.sh, but the source is an EXPORT of the working copy's HEAD commit and,
# recursively, of every submodule at the commit it pins, written from the git object stores (read-tree +
# checkout-index into a scratch index; the tuning campaign's export_commit method, tools/host.sh of
# sarc-1.5-b580-fused-port, copied) into <box>/.artifacts/src/<tag>/executorch. Uncommitted or untracked files of
# the live trees cannot reach the build. <tag>.export-manifest lists every exported tree with its commit.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")"); ET=$(cd $HERE/../../../.. && pwd); BOX=$(dirname $ET); A=$BOX/.artifacts; TAG=${1:?tag}
mkdir -p $A/build $A/src; P=$A/build/$TAG.src.txt; SRC=$A/src/$TAG/executorch; MAN=$A/src/$TAG.export-manifest
[[ -e $P || -e $SRC ]] && { echo "tag $TAG exists" >&2; exit 2; }
export_commit() { local wt=$1 sha=$2 dest=$3 rel=${4:-.} idx p s
  git -C "$wt" cat-file -e "$sha^{commit}" 2>/dev/null || { echo "export: $rel: commit $sha not available in $wt" >&2; return 1; }
  idx=$(mktemp -p $A/src) && rm -f $idx && mkdir -p "$dest" || return 1
  GIT_INDEX_FILE=$idx git -C "$wt" read-tree "$sha" && GIT_INDEX_FILE=$idx git -C "$wt" checkout-index -a -f --prefix="$dest/" \
    || { rm -f $idx; echo "export: $rel: checkout of $sha failed" >&2; return 1; }
  rm -f $idx; echo "$rel $sha" >> "$MAN"
  while read -r s p; do
    [[ -e $wt/$p/.git ]] || { echo "export: submodule $rel/$p is not initialised" >&2; return 1; }
    mkdir -p "$dest/$p" && export_commit "$wt/$p" "$s" "$dest/$p" "$rel/$p" || return 1
  done < <(git -C "$wt" ls-tree -r "$sha" | awk '$2 == "commit" {print $3, $4}'); }
exec 8>>$HOME/.cache/gpu-lab/lock-desktop-build; flock -x -w 14400 8 || { echo "build lock busy" >&2; exit 75; }
unset TMPDIR; IMAGE=${SARC_BUILD_IMAGE:-localhost/et-vk-build:rocky10}; SHA=$(git -C $ET rev-parse HEAD)
export_commit $ET $SHA $SRC || { echo "export failed" >&2; exit 2; }
{ echo "tag=$TAG"; echo "commit=$SHA"; echo "tree=$(git -C $ET rev-parse $SHA^{tree})"; echo "source=export $SRC"
  echo "export_manifest=$(wc -l < $MAN) trees sha256=$(sha256sum < $MAN | cut -c1-64)"
  echo "image=$IMAGE $(podman image inspect --format '{{.Id}}' $IMAGE)"; echo "start=$(date -u +%FT%TZ)"; } > $P
SARC_MOUNT_ROOT=$BOX $ET/sarc/tools/build.sh $SRC $A/build/$TAG > $A/build/$TAG.log 2>&1; rc=$?
grep -q SARC_BUILD_OK $A/build/$TAG.log || rc=${rc/#0/1}; echo "build_rc=$rc" >> $P
if [[ $rc == 0 ]]; then
  python3 $ET/sarc/tools/spirv_golden.py $A/build/$TAG/backend/vulkan_compute_shaders $ET/sarc/golden/spirv.json > $A/build/$TAG.golden.txt 2>&1
  echo "golden_rc=$? ($(tail -1 $A/build/$TAG.golden.txt))" >> $P; sha256sum $A/build/$TAG/tests/test_llama_microbench >> $P
fi
echo "end=$(date -u +%FT%TZ)" >> $P; tail -6 $P; exit $rc
