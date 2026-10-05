#!/bin/bash
# screen.sh <out name> <build tag> <scheme> <reps> <token...>: kernel-level screen with test_llama_microbench
# (--linear --regime=prefill, texture3d = the model path, all three models). Token "base" = no override (the
# release Xe2 row, which the B580 shares with the B70). Each token is run <reps> times, tokens interleaved; JSON per run in raw/<out name>/.
# This is a screen, not a gate: no correctness check here. A foreign GPU process (gl.sh status 76) ends the
# screen at once with SCREEN_ABORTED in env.txt and exit status 76; nothing further is launched.
. "$(dirname "$(readlink -f "$0")")/host.sh"; O=$A/raw/$1; B=$A/build/$2/tests/test_llama_microbench; Q=$3; R=$4; shift 4
mkdir -p $O; VAR=ET_VK_SARC_Q4GSW_VARIANT; [[ $Q == 8da4w ]] && VAR=ET_VK_SARC_DQ8CA_VARIANT
sha256sum $B > $O/env.txt; date -u >> $O/env.txt
for ((r = 1; r <= R; r++)); do for t in "$@"; do
  E=(); [[ $t != base ]] && E=("$VAR=$t")
  env "${E[@]}" $TOOLS/gl.sh $B --linear --regime=prefill --scheme=$Q --storage=texture3d --skip-correctness \
    --json-out=$O/$Q-$t-r$r.json > $O/$Q-$t-r$r.log 2>&1; rc=$?
  echo "$Q $t r$r rc=$rc temp=$(( $(gtemp_mc) / 1000 )) $(grep -o 'sarc_[a-z0-9_]*\|linear_[a-z0-9_]*tiled[a-z0-9_]*' $O/$Q-$t-r$r.log | sort | uniq -c | sort -rn | head -2 | tr '\n' ' ')"
  # 76 = a foreign GPU process, 75 = lock busy: stop measuring at once, keep the logs, report it
  [[ $rc == 76 || $rc == 75 ]] && { date -u >> $O/env.txt; echo "SCREEN_ABORTED rc=$rc at $Q $t r$r" | tee -a $O/env.txt; exit $rc; }
done; done
date -u >> $O/env.txt; echo SCREEN_DONE >> $O/env.txt
