#!/bin/bash
# stage.sh <session> <parent tag> "<parent env>" <cand tag> "<cand env>" [note]
# Host: stage/m51-LOCAL-ONLY/<session>/{parent,cand,parent-etdump,cand-etdump}/{llama_main,env,COMMIT}, the kit prompts
#   and r1329.txt, models-flat/ (verify.sh --flat-models names, symlinks to the model share), and for verify.sh the
#   wrappers llama_main / test_llama_microbench (-> adbshim.sh, running the candidate's binaries on the board).
# Board: $DEV_ROOT/stage/<session>/{parent,cand,parent-etdump,cand-etdump,top}: the same binaries and prompts; top =
#   the candidate's llama_main + test_llama_microbench (what verify.sh runs). STAGE.md lists every sha256, host and board.
# env strings are space-separated KEY=VALUE (may be empty).
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/dev.sh"
S=$ART/stage/$LOC/$1; PB=$2; PE=$3; CB=$4; CE=$5; NOTE=${6:-}; DS=$DEV_ROOT/stage/$1
[[ -e $S ]] && { echo "$S exists"; exit 1; }
ET=$(git -C "$TOOLS" rev-parse --show-toplevel); KIT=$ET/openspec/changes/sarc-1.5-e2e-benchmark/kit/prompts
PTE=${PTE_DIR:-<pte-dir>}
mkdir -p $S/{parent,cand,parent-etdump,cand-etdump,models-flat}
put() { # put <build dir> <dest> <env> <tag>
  cp -f $1/examples/models/llama/llama_main $2/
  : > $2/env; for kv in $3; do echo "$kv" >> $2/env; done
  sed -n 's/^tag \S* commit //p' $ART/build/m51/$4.txt > $2/COMMIT
}
put $ART/build/m51/$PB/llama $S/parent "$PE" $PB; put $ART/build/m51/$CB/llama $S/cand "$CE" $CB
put $ART/build/m51/$PB-etdump/llama-etdump $S/parent-etdump "$PE" $PB; put $ART/build/m51/$CB-etdump/llama-etdump $S/cand-etdump "$CE" $CB
cp -f $KIT/prompt_2048.txt $KIT/prompt_check.txt $TOOLS/r1329.txt $S/
cp -f $ART/build/m51/$CB/sarc_dev/test_llama_microbench $S/cand/
for p in llama_main test_llama_microbench; do printf '#!/bin/bash\nexec %s %s "$@"\n' "$TOOLS/adbshim.sh" $p > $S/$p; chmod +x $S/$p; done
for m in llama3_2-1b:llama3_2_1b llama3_2-3b:llama3_2_3b llama3_1-8b:llama3_1_8b; do for q in 4w 8da4w; do
  ln -s $PTE/${m#*:}_${q}_embq_ctx3072.pte $S/models-flat/${m%%:*}_vulkan_$q.pte; done; done
ln -s <tokenizer> $S/models-flat/tokenizer.model
A shell "mkdir -p $DS/parent $DS/cand $DS/parent-etdump $DS/cand-etdump $DS/top" < /dev/null
for d in parent cand parent-etdump cand-etdump; do A push $S/$d/llama_main $DS/$d/ > /dev/null; done
A push $S/cand/llama_main $DS/top/ > /dev/null; A push $S/cand/test_llama_microbench $DS/top/ > /dev/null
for f in prompt_2048.txt prompt_check.txt r1329.txt; do for d in parent cand parent-etdump cand-etdump top; do A push $S/$f $DS/$d/ > /dev/null; done; done
A shell "chmod 755 $DS/*/llama_main $DS/top/test_llama_microbench" < /dev/null
{ echo "session $1 staged $(date -u +%FT%TZ)"; echo "parent = build $PB env [$PE] commit $(cat $S/parent/COMMIT)"
  echo "cand   = build $CB env [$CE] commit $(cat $S/cand/COMMIT)"; echo "$NOTE"
  echo "host:"; (cd $S && sha256sum */llama_main cand/test_llama_microbench prompt_2048.txt prompt_check.txt r1329.txt)
  echo "board ($DS):"; A shell "cd $DS && sha256sum */llama_main top/test_llama_microbench top/*.txt" < /dev/null; } > $S/STAGE.md
cat $S/STAGE.md
