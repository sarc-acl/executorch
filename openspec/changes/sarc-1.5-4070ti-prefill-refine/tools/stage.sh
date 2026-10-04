#!/bin/bash
# stage.sh <session> <parent build tag> "<parent env>" <cand build tag> "<cand env>" [note]
# Stages a parent-vs-candidate session under stage/<session>/: {parent,cand}/{llama_main,libllama_runner.so,env,COMMIT},
# the traced binaries in {parent,cand}-traced/, the kit prompts and the unaligned prompt r1304.txt (if tools/r1304.txt exists), and (for verify.sh) the candidate's
# test_llama_microbench + llama_main at the top level. env strings are space-separated KEY=VALUE (may be empty).
set -euo pipefail
T=$(dirname "$0"); source "$T/common.sh"
S=$A/stage/$1; PB=$2; PE=$3; CB=$4; CE=$5; NOTE=${6:-}
mkdir -p $S/{parent,cand,parent-traced,cand-traced}
put() { # put <build tag> <dest> <env>
  local L=$A/build/$1/llama/examples/models/llama
  cp -f $L/llama_main $2/; cp -f $(find $A/build/$1/llama -name libllama_runner.so | head -1) $2/
  : > $2/env; for kv in $3; do echo "$kv" >> $2/env; done
  grep -v '^$' $A/build/$1.src.txt | sed -n '2p' > $2/COMMIT
}
put $PB $S/parent "$PE"; put $CB $S/cand "$CE"
for x in parent:$PB cand:$CB; do d=${x%%:*}; b=${x##*:}
  T=$A/build/$b-traced/llama/examples/models/llama
  [[ -x $T/llama_main ]] && { cp -f $T/llama_main $S/$d-traced/; cp -f $(find $A/build/$b-traced/llama -name libllama_runner.so | head -1) $S/$d-traced/; cp -f $S/$d/env $S/$d-traced/env; }
done
cp -f $KIT/prompts/prompt_*.txt $S/; [[ -f $T/r1304.txt ]] && cp -f $T/r1304.txt $S/
cp -f $S/cand/llama_main $S/cand/libllama_runner.so $S/
cp -f $A/build/$CB/tests/test_llama_microbench $S/ 2>/dev/null || echo "note: no test_llama_microbench in build/$CB/tests"
{ echo "session $1 staged $(date -u +%FT%TZ)"; echo "parent = build/$PB env [$PE] commit $(cat $S/parent/COMMIT)"
  echo "cand   = build/$CB env [$CE] commit $(cat $S/cand/COMMIT)"; echo "$NOTE"; } > $S/STAGE.md
sha256sum $S/parent/llama_main $S/cand/llama_main $S/test_llama_microbench 2>/dev/null >> $S/STAGE.md; cat $S/STAGE.md
