#!/bin/bash
# build-both.sh <tag> [tree]: container build of a tree (default: the working copy as it is) with
# sarc/tools/build.sh into build/<tag> (backend + tests + llama_main) and build/<tag>-traced (ETDump llama_main).
# build.sh calls podman; $A/bin/podman is the docker shim (podman-shim.sh).
set -uo pipefail
source "$(dirname "$0")/common.sh"; TAG=$1; T=${2:-$ET}
mkdir -p $A/build $A/bin; install -m 755 "$(dirname "$0")/podman-shim.sh" $A/bin/podman; export PATH=$A/bin:$PATH
{ date -u; git -C $T rev-parse HEAD; git -C $T status --short; } > $A/build/$TAG.src.txt 2>&1
$ET/sarc/tools/build.sh --llama $T $A/build/$TAG > $A/build/$TAG.log 2>&1; echo "rc=$? main" >> $A/build/$TAG.src.txt
$ET/sarc/tools/build.sh --llama --traced --no-tests $T $A/build/$TAG-traced > $A/build/$TAG-traced.log 2>&1; echo "rc=$? traced" >> $A/build/$TAG.src.txt
date -u >> $A/build/$TAG.src.txt; echo BUILD_BOTH_DONE >> $A/build/$TAG.src.txt
