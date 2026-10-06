#!/bin/bash
# build-both.sh <tag>: container build of ~/hmz-sarc/executorch (current working tree) with
# sarc/tools/build.sh into build/<tag> (backend + tests + llama_main) and build/<tag>-traced (ETDump llama_main).
set -uo pipefail
A=$HOME/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-03; ET=$HOME/hmz-sarc/executorch; TAG=$1
export SARC_MOUNT_ROOT=$HOME/hmz-sarc
{ date -u; git -C $ET rev-parse HEAD; git -C $ET status --short; } > $A/build/$TAG.src.txt 2>&1
"$(dirname "$(readlink -f "$0")")/hold.sh" run "build $TAG" $ET/sarc/tools/build.sh --llama $ET $A/build/$TAG > $A/build/$TAG.log 2>&1; echo "rc=$? main" >> $A/build/$TAG.src.txt
"$(dirname "$(readlink -f "$0")")/hold.sh" run "build $TAG-traced" $ET/sarc/tools/build.sh --llama --traced --no-tests $ET $A/build/$TAG-traced > $A/build/$TAG-traced.log 2>&1; echo "rc=$? traced" >> $A/build/$TAG.src.txt
date -u >> $A/build/$TAG.src.txt; echo BUILD_BOTH_DONE >> $A/build/$TAG.src.txt
