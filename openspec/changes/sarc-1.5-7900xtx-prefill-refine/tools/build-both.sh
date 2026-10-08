#!/bin/bash
# build-both.sh <commit> <tag>: export <commit> (export_commit.sh), then build it natively into build/7900xtx/<tag>
# (backend + tests + llama_main) and build/7900xtx/<tag>-traced (ETDump llama_main). One unit of the coordinator
# hold each. CPU work of the control workstation only; the GPU host builds nothing (R5).
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
C=$1; TAG=$2; S=$A/src/7900xtx/$TAG; O=$A/build/7900xtx/$TAG
[[ -e $O ]] && { echo "build $TAG exists: a rebuild takes a new tag" >&2; exit 2; }
$T/export_commit.sh $C $TAG > $A/build/7900xtx/$TAG.export.log 2>&1 || { echo "export failed" >&2; exit 3; }
{ date -u +%FT%TZ; echo "commit $(cat $S/COMMIT)"; echo "icd: AMDVLK 2025.Q2.1 on the GPU host (/etc/vulkan/icd.d/amd_icd64.json)"; } > $A/build/7900xtx/$TAG.src.txt
touch $A/.building
$T/hold.sh run "build $TAG" nice -n 10 $T/build-native.sh $S/executorch $O > $A/build/7900xtx/$TAG.log 2>&1; echo "rc=$? main" >> $A/build/7900xtx/$TAG.src.txt
$T/hold.sh run "build $TAG-traced" nice -n 10 $T/build-native.sh --traced --no-tests $S/executorch $O-traced > $A/build/7900xtx/$TAG-traced.log 2>&1; echo "rc=$? traced" >> $A/build/7900xtx/$TAG.src.txt
rm -f $A/.building
cp $S/MANIFEST $O/EXPORT-MANIFEST; date -u +%FT%TZ >> $A/build/7900xtx/$TAG.src.txt; echo BUILD_BOTH_DONE >> $A/build/7900xtx/$TAG.src.txt
