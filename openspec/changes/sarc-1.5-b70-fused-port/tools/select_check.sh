#!/bin/bash
# select_check.sh <name> <build tag A> <build tag B>: hook condition D4, test_sarc_select. Builds the two
# test_sarc_select executables of sarc/tools/check.sh (release tables only; with the dev zone) from the exported
# sources of each build (src/<tag>/executorch), runs them, keeps the executables and outputs under raw/<name>/
# and compares the release-table output of A and B. The dev-zone outputs differ when B adds dev-zone rows;
# they are kept for the record. No GPU.
. "$(dirname "$(readlink -f "$0")")/host.sh"; O=$A/raw/${1:?name}; mkdir -p $O; shift
for t in "$@"; do S=$A/src/$t/executorch; ( cd $S || exit 2; I=backends/vulkan/runtime/graph/ops/impl
  YAMLS=$(ls backends/vulkan/runtime/graph/ops/glsl/sarc/*.yaml backends/vulkan/runtime/graph/ops/glsl/sarc_dev/*.yaml); SEL=$(ls $I/sarc/Select.cpp $I/sarc/table_*.cpp)
  c++ -std=c++17 -Wall -I"$S/.." backends/vulkan/test/sarc_dev/test_sarc_select.cpp $SEL -o $O/test_sarc_select-rel-$t && $O/test_sarc_select-rel-$t $YAMLS > $O/select-rel-$t.txt 2>&1; echo "rel $t rc=$? $(tail -1 $O/select-rel-$t.txt)"
  c++ -std=c++17 -Wall -I"$S/.." backends/vulkan/test/sarc_dev/test_sarc_select.cpp $SEL $I/sarc_dev/*.cpp -o $O/test_sarc_select-dev-$t && ET_VK_SARC_UNVERIFIED=1 $O/test_sarc_select-dev-$t $YAMLS > $O/select-dev-$t.txt 2>&1; echo "dev $t rc=$? $(tail -1 $O/select-dev-$t.txt)" ); done
cmp $O/select-rel-$1.txt $O/select-rel-$2.txt && echo "SELECT_RELEASE_SAME ($1 vs $2)" || echo "SELECT_RELEASE_DIFFERENT"
sha256sum $O/test_sarc_select-* > $O/hashes.txt
