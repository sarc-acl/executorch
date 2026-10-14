#!/bin/bash
# build-micro.sh <commit> <tag>: exploration build for kernel-level work (round 2): export <commit> (export_commit.sh), build ONLY the Vulkan
# backend and the sarc_dev tests (test_llama_microbench) natively into build/rx7600/<tag> (no llama_main, no traced build). Same toolchain and
# cmake invocations as build-native.sh's backend/tests part. The result is for kernel timings and ISA statistics, never for a timed session.
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
C=$1; TAG=$2; S=$A/src/rx7600/$TAG; O=$A/build/rx7600/$TAG
[[ -e $O || -L $O ]] && { echo "build $TAG exists: a rebuild takes a new tag" >&2; exit 2; }
if [[ -n ${SARC_BIG:-} ]]; then mkdir -p $SARC_BIG/build; [[ -e $SARC_BIG/build/$TAG ]] && { echo "build $TAG exists on $SARC_BIG" >&2; exit 2; }; mkdir -p $SARC_BIG/build/$TAG; ln -s $SARC_BIG/build/$TAG $O; fi
$T/export_commit.sh $C $TAG > $A/build/rx7600/$TAG.export.log 2>&1 || { echo "export failed" >&2; exit 3; }
{ date -u +%FT%TZ; echo "commit $(cat $S/COMMIT) (micro build: backend + tests only)"; echo "mesa: $VK_ICD_FILENAMES (Mesa 26.2.3 31e9a6b2e9)"; } > $A/build/rx7600/$TAG.src.txt
touch $A/.building
SRC=$S/executorch; mkdir -p $O; cd $SRC
export PYTHONPATH=$(dirname "$SRC") CCACHE_DIR=$A/ccache
PY=<toolchain-share>/Python-3.12.9-1/bin/python3; GL=<vulkan-sdk>/1.4.350.1/x86_64/bin/glslc; JOBS=${SARC_JOBS:-24}
CC=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
B=$O/backend; TT=$O/tests
{ $T/hold.sh run "build $TAG" bash -c "cmake . -DCMAKE_INSTALL_PREFIX=$B -DEXECUTORCH_BUILD_VULKAN=ON -DPYTHON_EXECUTABLE=$PY ${CC[*]} -DGLSLC_PATH=$GL -B$B && cmake --build $B -j$JOBS --target install && cmake backends/vulkan/test/sarc_dev -DCMAKE_PREFIX_PATH=$B -DCMAKE_FIND_ROOT_PATH=$B -DCMAKE_BUILD_TYPE=Debug -DEXECUTORCH_ROOT=$SRC ${CC[*]} -B$TT && cmake --build $TT -j$JOBS && echo SARC_MICRO_BUILD_OK"; } > $A/build/rx7600/$TAG.log 2>&1
echo "rc=$?" >> $A/build/rx7600/$TAG.src.txt
rm -f $A/.building; cp $S/MANIFEST $O/EXPORT-MANIFEST; date -u +%FT%TZ >> $A/build/rx7600/$TAG.src.txt; echo BUILD_MICRO_DONE >> $A/build/rx7600/$TAG.src.txt
