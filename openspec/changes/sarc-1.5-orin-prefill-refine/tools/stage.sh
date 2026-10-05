#!/bin/bash
# stage.sh <session> <parent build tag> "<parent env>" <cand build tag> "<cand env>" [note]
# Device side (after deploy.sh has copied the bundles). Stages a parent-vs-candidate session under stage/<session>/:
# {parent,cand}/{llama_main,libllama_runner.so,env,COMMIT}, the same binaries again in {parent,cand}-traced/ (the
# cross build links ETDump into the one runner; it traces only with --etdump_path), the kit prompts, the unaligned
# prompt r1304.txt (unchanged from the earlier campaigns, sha256 881de104...) and, for verify.sh, the candidate's test_llama_microbench +
# llama_main at the top level. env strings are space-separated KEY=VALUE (may be empty).
# Nothing is staged unless every binary and prompt exists (exit 77).
set -euo pipefail
source "$(dirname "$0")/common.sh"
[[ $SIDE == device ]] || { echo "device only" >&2; exit 2; }
S=$A/stage/$1; PB=$2; PE=$3; CB=$4; CE=$5; NOTE=${6:-}
R1304=881de104b6b04bba7a40070283a1a27d9c5e5b3988af8ae14704ce388b5f139a
bins() { echo "$A/build/${1%-traced}/bundle/llama_main $A/build/${1%-traced}/bundle/libllama_runner.so"; }
for b in $PB $CB $PB-traced $CB-traced; do
  read -r M L <<< "$(bins $b)"; need "$M" "${L:-$A/build/$b/libllama_runner.so}" $A/build/${b%-traced}.src.txt; done
need $A/build/$CB/bundle/test_llama_microbench $KIT/prompts/prompt_2048.txt $KIT/prompts/prompt_check.txt \
  $KIT/prompts/prompt_real_2048.txt $TOOLS/r1304.txt
[[ $(sha256sum < $TOOLS/r1304.txt | cut -d' ' -f1) == "$R1304" ]] || { echo "r1304.txt changed" >&2; exit 77; }
[[ -e $S/raw/runs.csv ]] && { echo "session $1 already has results; use a new session name" >&2; exit 77; }
mkdir -p $S/{parent,cand,parent-traced,cand-traced}
put() { # put <build tag> <dest> <env>
  local M L; read -r M L <<< "$(bins $1)"; cp -f $M $L $2/
  : > $2/env; for kv in $3; do echo "$kv" >> $2/env; done
  grep '^commit ' $A/build/${1%-traced}.src.txt | head -1 | cut -d' ' -f2 > $2/COMMIT
}
put $PB $S/parent "$PE"; put $CB $S/cand "$CE"; put $PB-traced $S/parent-traced "$PE"; put $CB-traced $S/cand-traced "$CE"
cp -f $KIT/prompts/prompt_*.txt $TOOLS/r1304.txt $S/
# verify.sh runs ./llama_main: that is the exit-status wrapper, the candidate's runner is in verify-bin/.
source $TOOLS/gatelib.sh; stage_verify_runner $S $S/cand/llama_main
cp -f $S/cand/libllama_runner.so $A/build/$CB/bundle/test_llama_microbench $S/
{ echo "session $1 staged $(date -u +%FT%TZ)"; echo "parent = build/$PB env [$PE] commit $(cat $S/parent/COMMIT)"
  echo "cand   = build/$CB env [$CE] commit $(cat $S/cand/COMMIT)"; echo "$NOTE"
  grep -h '^local-patch\|^tree-sha256\|^source ' $A/build/$PB.src.txt $A/build/$CB.src.txt
  sha256sum $S/parent/* $S/cand/* $S/parent-traced/llama_main $S/cand-traced/llama_main $S/verify-bin/llama_main $S/llama_main $S/test_llama_microbench $S/*.txt; } > $S/STAGE.md
cat $S/STAGE.md
