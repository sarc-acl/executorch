#!/bin/bash
# stage.sh <session> <parent build tag> "<parent env>" <cand build tag> "<cand env>" [note]
# Stages a parent-vs-candidate session under stage/<session>/: {parent,cand}/{llama_main,libllama_runner.so,env,COMMIT},
# the traced binaries in {parent,cand}-traced/, the kit prompts, and (for verify.sh) the candidate's
# test_llama_microbench + llama_main + libllama_runner.so at the top level. env strings are space-separated
# KEY=VALUE (may be empty); ET_VK_SARC_UNVERIFIED=1 is part of both arms (the RX 7600 rows are kUnverified).
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$A/stage/$1; PB=$2; PE=$3; CB=$4; CE=$5; NOTE=${6:-}
[[ -e $S ]] && { echo "stage $1 exists" >&2; exit 2; }
mkdir -p $S/{parent,cand,parent-traced,cand-traced}
put() { # put <build tag> <dest> <env>
  local L=$A/build/rx7600/$1/llama/examples/models/llama
  cp -f $L/llama_main $2/; cp -f $(find $A/build/rx7600/$1/llama -name libllama_runner.so | head -1) $2/
  : > $2/env; for kv in $3; do echo "$kv" >> $2/env; done
  cat $A/src/rx7600/$1/COMMIT > $2/COMMIT
}
put $PB $S/parent "$PE"; put $CB $S/cand "$CE"
for x in parent:$PB cand:$CB; do d=${x%%:*}; b=${x##*:}
  TL=$A/build/rx7600/$b-traced/llama/examples/models/llama
  [[ -x $TL/llama_main ]] && { cp -f $TL/llama_main $S/$d-traced/; cp -f $(find $A/build/rx7600/$b-traced/llama -name libllama_runner.so | head -1) $S/$d-traced/; cp -f $S/$d/env $S/$d-traced/env; }
done
cp -f $ET/openspec/changes/sarc-1.5-e2e-benchmark/kit/prompts/prompt_*.txt $S/
[[ -f $T/r1304.txt ]] && cp -f $T/r1304.txt $S/
cp -f $S/cand/llama_main $S/cand/libllama_runner.so $S/
cp -f $A/build/rx7600/$CB/tests/test_llama_microbench $S/ 2>/dev/null || echo "note: no test_llama_microbench in build/rx7600/$CB/tests"
{ echo "session $1 staged $(date -u +%FT%TZ)"; echo "parent = build/rx7600/$PB env [$PE] commit $(cat $S/parent/COMMIT)"
  echo "cand   = build/rx7600/$CB env [$CE] commit $(cat $S/cand/COMMIT)"; echo "driver = $VK_ICD_FILENAMES (Mesa 26.2.3 31e9a6b2e9)"; echo "$NOTE"; } > $S/STAGE.md
sha256sum $S/parent/llama_main $S/cand/llama_main $S/parent/libllama_runner.so $S/cand/libllama_runner.so $S/test_llama_microbench 2>/dev/null >> $S/STAGE.md; cat $S/STAGE.md
