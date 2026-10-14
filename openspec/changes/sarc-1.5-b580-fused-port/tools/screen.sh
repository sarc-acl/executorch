#!/bin/bash
# screen.sh <out name> <build tag> <scheme> <reps> <token...>: kernel-level screen with test_llama_microbench
# (--linear --regime=prefill, texture3d = the model path, all three models). Token "base" = no override (the
# release Xe2 row, which the B580 shares with the B70). Each token is run <reps> times, tokens interleaved.
# Resumable: raw/<out name>/rows.csv gets one row per (token, repeat, shape) and a (token, repeat) whose rows
# are all saved is skipped. Rows missing from the CSV are recovered from the run's cached JSON when that is a
# valid, complete result; only then is nothing launched. A failed or interrupted attempt is moved to
# raw/<out name>/superseded/<reason>/ before its run is repeated; no log, JSON or CSV row is ever overwritten
# (tools/screen_rows.py holds the rules). env.txt is appended to: one block per invocation.
# B580_SCREEN_RECOVER_ONLY=1: recover rows from cached results and report what would run; launches nothing.
# Each run starts from a cooled card (host.sh cool_start). A screen, not a gate: no correctness check here.
# A foreign GPU workload (gl.sh status 76) or a busy lock (75) ends the screen at once with SCREEN_ABORTED in
# env.txt and that exit status. Exit status 1 and SCREEN_INCOMPLETE if a run left no valid result.
. "$(dirname "$(readlink -f "$0")")/host.sh"; N=$1; O=$A/raw/$1; B=$A/build/$2/tests/test_llama_microbench; Q=$3; R=$4; shift 4
mkdir -p $O; VAR=ET_VK_SARC_Q4GSW_VARIANT; [[ $Q == 8da4w ]] && VAR=ET_VK_SARC_DQ8CA_VARIANT
DRY=${B580_SCREEN_RECOVER_ONLY:+--no-supersede}; bad=0; todo=0
{ echo "== $(date -u +%FT%TZ) screen.sh $N scheme=$Q reps=$R${DRY:+ RECOVER_ONLY} tokens: $*"; sha256sum $B; } >> $O/env.txt
for ((r = 1; r <= R; r++)); do for t in "$@"; do
  S=$O/$Q-$t-r$r
  python3 $TOOLS/screen_rows.py linear $O $Q $t $r $DRY; st=$?
  [[ $st == 0 ]] && continue
  [[ $st == 10 ]] || { echo "SCREEN_FAILED screen_rows.py status $st at $Q $t r$r" | tee -a $O/env.txt; exit 2; }
  [[ -n $DRY ]] && { echo "WOULD_RUN $Q $t r$r"; todo=$((todo + 1)); continue; }
  E=(); [[ $t != base ]] && E=("$VAR=$t")
  cool_start   # B580: without it the first token after a long job runs hot (screen3-4w, base round 1: 17 to 21 % slow)
  date -u +%FT%TZ > $S.started
  env "${E[@]}" $TOOLS/gl.sh $B --linear --regime=prefill --scheme=$Q --storage=texture3d --skip-correctness \
    --json-out=$S.json > $S.log 2>&1; rc=$?
  echo "$rc $(( $(gtemp_mc) / 1000 ))" > $S.rc
  echo "$Q $t r$r rc=$rc temp=$(( $(gtemp_mc) / 1000 )) $(grep -o 'sarc_[a-z0-9_]*\|linear_[a-z0-9_]*tiled[a-z0-9_]*' $S.log | sort | uniq -c | sort -rn | head -2 | tr '\n' ' ')"
  # 76 = a foreign GPU workload, 75 = lock busy: stop measuring at once; the attempt stays and is superseded on resume
  [[ $rc == 76 || $rc == 75 ]] && { echo "$(date -u +%FT%TZ) SCREEN_ABORTED rc=$rc at $Q $t r$r" | tee -a $O/env.txt; exit $rc; }
  python3 $TOOLS/screen_rows.py linear $O $Q $t $r --no-supersede || { bad=$((bad + 1)); echo "$(date -u +%FT%TZ) no valid result: $Q $t r$r rc=$rc" | tee -a $O/env.txt; }
done; done
[[ -n $DRY ]] && { echo "$(date -u +%FT%TZ) RECOVER_ONLY done, $todo run(s) not saved" | tee -a $O/env.txt; exit 0; }
[[ $bad == 0 ]] || { echo "$(date -u +%FT%TZ) SCREEN_INCOMPLETE $bad run(s) without a valid result" | tee -a $O/env.txt; exit 1; }
echo "$(date -u +%FT%TZ) SCREEN_DONE" >> $O/env.txt
