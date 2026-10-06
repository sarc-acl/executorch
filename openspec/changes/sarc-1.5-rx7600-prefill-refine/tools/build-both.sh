#!/bin/bash
# build-both.sh <commit> <tag>: export <commit> (export_commit.sh), then build it natively into build/rx7600/<tag>
# (backend + tests + llama_main) and build/rx7600/<tag>-traced (ETDump llama_main). One unit of the coordinator
# hold each. CPU work: never during a timed session (e2e5.sh refuses to start while $A/.building exists).
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
C=$1; TAG=$2; S=$A/src/rx7600/$TAG; O=$A/build/rx7600/$TAG
[[ -e $O ]] && { echo "build $TAG exists: a rebuild takes a new tag" >&2; exit 2; }
$T/export_commit.sh $C $TAG > $A/build/rx7600/$TAG.export.log 2>&1 || { echo "export failed" >&2; exit 3; }
{ date -u +%FT%TZ; echo "commit $(cat $S/COMMIT)"; echo "mesa: $VK_ICD_FILENAMES (Mesa 26.2.3 31e9a6b2e9)"; } > $A/build/rx7600/$TAG.src.txt
touch $A/.building
$T/hold.sh run "build $TAG" $T/build-native.sh $S/executorch $O > $A/build/rx7600/$TAG.log 2>&1; echo "rc=$? main" >> $A/build/rx7600/$TAG.src.txt
$T/hold.sh run "build $TAG-traced" $T/build-native.sh --traced --no-tests $S/executorch $O-traced > $A/build/rx7600/$TAG-traced.log 2>&1; echo "rc=$? traced" >> $A/build/rx7600/$TAG.src.txt
rm -f $A/.building
cp $S/MANIFEST $O/EXPORT-MANIFEST; date -u +%FT%TZ >> $A/build/rx7600/$TAG.src.txt; echo BUILD_BOTH_DONE >> $A/build/rx7600/$TAG.src.txt
