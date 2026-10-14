#!/bin/bash
# screen_sdpa.sh <out name> <build tag> <reps> <profile...>: kernel-level SDPA screen with test_llama_microbench
# --sdpa (prefill S = 2048 on a 3072 cache, the 1B / 3B / 8B head configurations; QK^T, softmax and attn*V
# timed separately). Profile "base" = no environment (the stock kernels the parent runs); any other name is run
# as ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=<profile>. Profiles interleaved, <reps> rounds; one log per
# run in raw/<out name>/ and one row per (profile, round, model, sub-op, variant) in raw/<out name>/screen.csv.
# Resumable by saved row keys, not by the status marker: a (profile, round) is skipped when all its rows are
# in the CSV; rows that are missing (the screen was interrupted between the run and the append) are recovered
# from the run's log when that log is a valid, complete result; otherwise the earlier attempt is moved to
# raw/<out name>/superseded/<reason>/ and the run is repeated (tools/screen_rows.py holds the rules). Nothing
# is overwritten; env.txt is appended to. B580_SCREEN_RECOVER_ONLY=1 launches nothing.
# A screen, not a gate: no correctness check. A foreign GPU workload (gl.sh status 76) or a busy lock (75)
# ends it with SCREEN_ABORTED. Exit status 1 and SCREEN_INCOMPLETE if a run left no valid result.
. "$(dirname "$(readlink -f "$0")")/host.sh"; N=$1; O=$A/raw/$1; B=$A/build/$2/tests/test_llama_microbench; R=$3; shift 3
mkdir -p $O; DRY=${B580_SCREEN_RECOVER_ONLY:+--no-supersede}; bad=0; todo=0
{ echo "== $(date -u +%FT%TZ) screen_sdpa.sh $N reps=$R${DRY:+ RECOVER_ONLY} profiles: $*"; sha256sum $B; } >> $O/env.txt
for ((r = 1; r <= R; r++)); do for p in "$@"; do
  L=$O/$p-r$r.log
  python3 $TOOLS/screen_rows.py sdpa $O $p $r $DRY; st=$?
  [[ $st == 0 ]] && continue
  [[ $st == 10 ]] || { echo "SCREEN_FAILED screen_rows.py status $st at $p r$r" | tee -a $O/env.txt; exit 2; }
  [[ -n $DRY ]] && { echo "WOULD_RUN $p r$r"; todo=$((todo + 1)); continue; }
  E=(); [[ $p != base ]] && E=(ET_VK_SARC_UNVERIFIED=1 "ET_VK_SARC_DEV_PROFILE=${p%%+*}")
  [[ $p == *+softmax* ]] && E+=("ET_VK_SARC_XE2_SOFTMAX=${p##*+softmax}")   # "<profile>+softmax1|2": hook builds only (tools/build-hook.sh)
  idle_wait "screen $N $p r$r"; cool_start
  date -u +%FT%TZ > $L.started
  env "${E[@]}" $TOOLS/gl.sh $B --sdpa > $L 2>&1; rc=$?; t=$(( $(gtemp_mc) / 1000 )); echo "$rc $t" > $L.rc
  echo "$p r$r rc=$rc temp=$t $(grep -o 'sarc_sdpa_[a-z0-9_]*' $L | sort -u | tr '\n' ' ')"
  [[ $rc == 76 || $rc == 75 ]] && { echo "$(date -u +%FT%TZ) SCREEN_ABORTED rc=$rc at $p r$r" | tee -a $O/env.txt; exit $rc; }
  python3 $TOOLS/screen_rows.py sdpa $O $p $r --no-supersede || { bad=$((bad + 1)); echo "$(date -u +%FT%TZ) no valid result: $p r$r rc=$rc" | tee -a $O/env.txt; }
done; done
[[ -n $DRY ]] && { echo "$(date -u +%FT%TZ) RECOVER_ONLY done, $todo run(s) not saved" | tee -a $O/env.txt; exit 0; }
[[ $bad == 0 ]] || { echo "$(date -u +%FT%TZ) SCREEN_INCOMPLETE $bad run(s) without a valid result" | tee -a $O/env.txt; exit 1; }
echo "$(date -u +%FT%TZ) SCREEN_DONE" >> $O/env.txt
