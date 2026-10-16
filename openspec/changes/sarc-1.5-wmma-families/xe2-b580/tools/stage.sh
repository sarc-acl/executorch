#!/bin/bash
# stage.sh <session> <parent build tag> "<parent env>" <cand build tag> "<cand env>" [note]
# Stages a parent-vs-candidate session under stage/<session>/: {parent,cand}/{llama_main,libllama_runner.so,env,COMMIT},
# the traced binaries in {parent,cand}-traced/, the kit prompts, the unaligned prompt r1304.txt that
# sarc/tools/verify.sh looks for, and (for verify.sh) the candidate's test_llama_microbench + llama_main at the
# top level. env strings are space-separated KEY=VALUE (may be empty). Only builds that build-both.sh marked
# BUILD_BOTH_OK can be staged.
set -euo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"
S=$A/stage/$1; PB=$2; PE=$3; CB=$4; CE=$5; NOTE=${6:-}
UNALIGNED=${B580_UNALIGNED:-$HOME/.cache/et-e2e/sarc15-r4/r1304.txt}   # sha256 881de104b6b0..., as in the earlier B70 studies
for b in $PB $CB; do grep -qx BUILD_BOTH_OK $A/build/$b.src.txt || { echo "build/$b is not a successful build" >&2; exit 2; }; done
[[ -f $UNALIGNED ]] || { echo "missing unaligned prompt $UNALIGNED" >&2; exit 2; }
[[ -e $S ]] && { echo "session $S already exists" >&2; exit 2; }
mkdir -p $S/{parent,cand,parent-traced,cand-traced}
put() { # put <build tag> <dest> <env>
  local L=$A/build/$1/llama/examples/models/llama
  python3 "$TOOLS/check_build_hashes.py" "$A/build/$1.src.txt" "$A/build/$1" "$A/build/$1-traced" || return 2
  cp -f $L/llama_main $2/llama_main.bin; cp -f "$TOOLS/launch_runner.py" $2/llama_main; chmod +x $2/llama_main; cp -f $(find $A/build/$1/llama -name libllama_runner.so | head -1) $2/
  : > $2/env; for kv in $3; do echo "$kv" >> $2/env; done
  sed -n 's/^commit=//p' $A/build/$1.src.txt > $2/COMMIT
}
put $PB $S/parent "$PE"; put $CB $S/cand "$CE"
for x in parent:$PB cand:$CB; do d=${x%%:*}; b=${x##*:}
  TB=$A/build/$b-traced/llama/examples/models/llama
  cp -f $TB/llama_main $(find $A/build/$b-traced/llama -name libllama_runner.so | head -1) $S/$d-traced/; cp -f $S/$d/env $S/$d-traced/env
done
cp -f $ET/openspec/changes/sarc-1.5-e2e-benchmark/kit/prompts/prompt_*.txt $UNALIGNED $S/
cp -f $S/cand/llama_main $S/cand/llama_main.bin $S/cand/libllama_runner.so $S/cand/env $A/build/$CB/tests/test_llama_microbench $S/
{ echo "session $1 staged $(date -u +%FT%TZ)"; echo "parent = build/$PB env [$PE] commit $(cat $S/parent/COMMIT)"
  echo "cand   = build/$CB env [$CE] commit $(cat $S/cand/COMMIT)"; echo "$NOTE"
  find "$S" -maxdepth 2 -type f \( -name llama_main -o -name llama_main.bin -o -name libllama_runner.so -o -name test_llama_microbench -o -name env -o -name COMMIT -o -name "*.txt" \) -exec sha256sum {} +; } > $S/STAGE.md
cat $S/STAGE.md
