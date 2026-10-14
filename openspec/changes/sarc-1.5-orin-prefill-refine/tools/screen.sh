#!/bin/bash
# screen.sh <out name> <build tag> <scheme> <reps> <token...>: kernel-level screen with test_llama_microbench
# (--linear --regime=prefill, texture3d = the model path, all three models). Device side. Token "base" = no override
# (the release Orin rows). Each token is run <reps> times, tokens interleaved; JSON per run in raw/<out name>/.
# Resumable: a (token, repeat) whose JSON exists is skipped. This is a screen, not a gate: no correctness check.
T=$(dirname "$0"); source "$T/common.sh"; O=$A/raw/$1; BD=$A/build/$2/bundle; B=$BD/test_llama_microbench; Q=$3; R=$4; shift 4
[[ $SIDE == device ]] || { echo "device only" >&2; exit 2; }
need $B; mkdir -p $O; VAR=ET_VK_SARC_Q4GSW_VARIANT; [[ $Q == 8da4w ]] && VAR=ET_VK_SARC_DQ8CA_VARIANT
sha256sum $B > $O/env.txt; date -u >> $O/env.txt
for ((r = 1; r <= R; r++)); do for t in "$@"; do
  [[ -s $O/$Q-$t-r$r.json ]] && continue
  E=(); [[ $t != base ]] && E=("$VAR=$t")
  cool_start 120
  env "${E[@]}" LD_LIBRARY_PATH=$BD $T/gl.sh $B --linear --regime=prefill --scheme=$Q --storage=texture3d --skip-correctness \
    --json-out=$O/$Q-$t-r$r.json > $O/$Q-$t-r$r.log 2>&1
  rc=$?; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "screen stopped rc=$rc (70 device lost, 75 lock busy, 76 foreign GPU process)"; exit $rc; }
  echo "$Q $t r$r rc=$rc temp=$(gtemp) $(grep -o 'sarc_[a-z0-9_]*\|linear_[a-z0-9_]*tiled[a-z0-9_]*' $O/$Q-$t-r$r.log | sort | uniq -c | sort -rn | head -2 | tr '\n' ' ')"
done; done
date -u >> $O/env.txt; echo SCREEN_DONE >> $O/env.txt
