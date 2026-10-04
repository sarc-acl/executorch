#!/bin/bash
# screen_sdpa.sh <out name> <build tag> <reps> <profile...>: kernel-level SDPA screen with test_llama_microbench
# --sdpa (prefill S = 2048 on a 3072 cache, the 1B / 3B / 8B head configurations; QK^T, softmax and attn*V
# timed separately). Profile "base" = no environment (the stock kernels the parent runs); any other name is run
# as ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=<profile>. Profiles interleaved, <reps> rounds; one log per
# run in raw/<out name>/ and one row per (profile, round, model, sub-op) appended to raw/<out name>/screen.csv.
# Resumable: a (profile, round) whose log ended with rc 0 is skipped. A screen, not a gate: no correctness
# check. A foreign GPU process (gl.sh status 76) or a busy lock (75) ends it with SCREEN_ABORTED.
. "$(dirname "$(readlink -f "$0")")/host.sh"; O=$A/raw/$1; B=$A/build/$2/tests/test_llama_microbench; R=$3; shift 3
mkdir -p $O; CSV=$O/screen.csv
[[ -f $CSV ]] || echo "profile,rep,model,op,variant,mean_us,stdev_us,dispatch,temp_c,rc" > $CSV
{ sha256sum $B; date -u; } >> $O/env.txt
for ((r = 1; r <= R; r++)); do for p in "$@"; do
  L=$O/$p-r$r.log; [[ -f $L.rc && ( $(<$L.rc) == 0 || ( $p == base && $(<$L.rc) == 1 ) ) ]] && continue
  E=(); [[ $p != base ]] && E=(ET_VK_SARC_UNVERIFIED=1 "ET_VK_SARC_DEV_PROFILE=$p")
  cool_start
  env "${E[@]}" $TOOLS/gl.sh $B --sdpa > $L 2>&1; rc=$?; echo $rc > $L.rc; t=$(( $(gtemp_mc) / 1000 ))
  # RESULT,sdpa,<model>,<scheme>,<regime>,<op>,<K>,<N>,<mean_us>,<stdev_us>,-1,<dispatch>,SKIPPED,<kv>,<variant>
  awk -F, -v p=$p -v r=$r -v t=$t -v rc=$rc '$1 == "RESULT" && $2 == "sdpa" && $5 == "prefill" {print p "," r "," $3 "," $6 "," $15 "," $9 "," $10 "," $12 "," t "," rc}' $L >> $CSV
  echo "$p r$r rc=$rc temp=$t $(grep -o 'sarc_sdpa_[a-z0-9_]*' $L | sort -u | tr '\n' ' ')"
  [[ $rc == 76 || $rc == 75 ]] && { date -u >> $O/env.txt; echo "SCREEN_ABORTED rc=$rc at $p r$r" | tee -a $O/env.txt; exit $rc; }
done; done
date -u >> $O/env.txt; echo SCREEN_DONE >> $O/env.txt
