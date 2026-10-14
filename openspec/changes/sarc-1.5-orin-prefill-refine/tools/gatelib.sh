#!/bin/bash
# gatelib.sh: shared steps of gate.sh, gate_sdpa.sh and parent_verify.sh (sourced after common.sh, with D = the
# stage directory). A failed step ends the gate: no further GPU job runs and gate.done records the step.
# Device loss (exit 70), a busy lock (75) and a foreign GPU process (76) are passed up unchanged as GATE_ABORTED.
PARENT_CTL=$A/stage/s0-parent-verify
# The per-cell normal-clock thresholds (calibrate_clock.py on the baseline and A/A sessions). Candidate gates
# refuse to start without it and e2e5.sh applies it; a record-only session cannot pass gate_check.py.
CLKFILE=$CHANGE/results/orin/clkmin.json
# stage_verify_runner <stage dir> <real llama_main>: verify.sh calls ./llama_main; stage the exit-status wrapper
# under that name and the real runner beside it, so that every call's status is recorded without touching verify.sh.
stage_verify_runner() { mkdir -p $1/verify-bin; cp -f $2 $1/verify-bin/llama_main; install -m 755 $TOOLS/llama_main_rc.sh $1/llama_main; }
# cand_env [given]: the ONE candidate environment of a session is the staged cand/env file (what e2e5.sh and
# trace.sh read). It must equal cand-traced/env; an environment given on the command line must be the same.
cand_env() {
  need $D/STAGE.md; [[ -f $D/cand/env && -f $D/cand-traced/env && -f $D/parent/env && -f $D/parent-traced/env ]] || { echo "session not staged with env files" >&2; exit 77; }
  cmp -s $D/cand/env $D/cand-traced/env && cmp -s $D/parent/env $D/parent-traced/env || { echo "env of the timed and traced binaries differ" >&2; exit 77; }
  ENVS=$(grep -v '^$' $D/cand/env | tr '\n' ' '); ENVS=${ENVS% }
  if [[ $# -gt 0 ]]; then local g; g=$(echo $1); [[ $g == "$ENVS" ]] || { echo "environment given ($g) is not the staged cand/env ($ENVS)" >&2; exit 77; }; fi
  echo "candidate environment: [$ENVS] sha256 $(sha256sum < $D/cand/env | cut -c1-16)" | tee $D/gate.env
}
# NEAR_TIE=<NEAR_TIE.json> (owner decision 2026-10-04): a DIFFER on a next-token item is then judged by
# gate_check.py against that evidence instead of rejecting by itself; an acceptance that used it is recorded as
# "ACCEPTED (near-tie, owner decision 2026-10-04)" with the evidence path, never as a plain pass.
# REF_ERROR=<REFERENCE_ERROR.json> (second owner decision 2026-10-04, for candidates that change kernel
# arithmetic): the same, judged by ref_error_rule.py's criteria and recorded as
# "ACCEPTED (reference-error rule, owner decision 2026-10-04)".
near_tie_arg() { NT=""; if [[ -n ${NEAR_TIE:-} ]]; then need "$NEAR_TIE"; NT="--near-tie $(realpath "$NEAR_TIE")"; fi
  if [[ -n ${REF_ERROR:-} ]]; then need "$REF_ERROR"; NT="$NT --reference-error $(realpath "$REF_ERROR")"; fi; }
accepted() {
  local items; items=$(grep -hs '^NEAR-TIE:' $D/verify-check.txt $D/session-check.txt | sed 's/^NEAR-TIE: //' | tr '\n' ';')
  if grep -qs 'ACCEPT (reference-error rule, owner decision 2026-10-04)' $D/verify-check.txt $D/session-check.txt; then
    finish GATE_ACCEPTED "ACCEPTED (reference-error rule, owner decision 2026-10-04) evidence $(realpath "$REF_ERROR"); differing items: $items the gain is in raw/summary.csv" 0
  fi
  if grep -qs 'ACCEPT (near-tie, owner decision 2026-10-04)' $D/verify-check.txt $D/session-check.txt; then
    finish GATE_ACCEPTED "ACCEPTED (near-tie, owner decision 2026-10-04) evidence $(realpath "$NEAR_TIE"); differing items: $items the gain is in raw/summary.csv" 0
  fi
  finish GATE_ACCEPTED "all steps passed; the gain is in raw/summary.csv" 0
}
finish() { echo "$1 $(date -u +%FT%TZ) $2" | tee $D/gate.done; exit $3; }
step() { # step <name> <command...>
  local name=$1; shift; "$@"; local rc=$?
  [[ $rc == 70 || -f $GONE ]] && finish GATE_ABORTED "device lost during $name" 70
  [[ $rc == 75 || $rc == 76 ]] && finish GATE_ABORTED "$name rc=$rc (75 lock busy, 76 foreign GPU process)" $rc
  gpu_alive || gpu_gone "after $name"
  [[ $rc == 0 ]] || finish GATE_REJECTED "step $name failed rc=$rc" $rc
}
run_verify() { # run_verify "<env>": the unmodified sarc/tools/verify.sh (deploy.sh copies it and checks its hash against the commit)
  need $D/llama_main $D/verify-bin/llama_main $D/libllama_runner.so $D/test_llama_microbench $D/prompt_2048.txt $D/prompt_check.txt $D/r1304.txt
  cmp -s $D/llama_main $TOOLS/llama_main_rc.sh || { echo "$D/llama_main is not the exit-status wrapper" >&2; return 77; }
  [[ -e $D/verify-runs.jsonl || -e $D/verify.out ]] && { echo "verify.sh already ran in $D; results are kept, use a new session" >&2; return 77; }
  # verify.sh is not modified: foreign GPU processes are watched from outside for its whole duration.
  no_others "before verify.sh"; others_watch_start $D/verify.others
  env $1 $ET/sarc/tools/verify.sh --dir $D --lock $LOCK --models 1b,3b,8b --schemes 4w,8da4w --pdiff --out verify --flat-models $MODELDIR > $D/verify.out 2>&1
  local rc=$?; echo "VERIFY_DONE rc=$rc" >> $D/verify.out; local o; o=$(others_watch_stop $D/verify.others)
  [[ -n $o ]] && { echo "foreign GPU process during verify.sh: $o" | tee -a $D/verify.out >> $A/ABORTED; return 76; }
  return $rc
}
timed_and_traced() { # the e2e session, its content check, then the warm traces
  step session $TOOLS/session.sh $(basename $D) --clkmin-file $CLKFILE > $D/e2e5.out 2>&1
  python3 $TOOLS/summarize.py $D/raw > $D/raw/summary.csv 2>&1
  step session-check python3 $TOOLS/gate_check.py session $D/raw --clkmin $CLKFILE --require-logs ${NT:-} > $D/session-check.txt 2>&1
  step trace $TOOLS/trace.sh $(basename $D) 1b,3b,8b 4w,8da4w "parent cand" > $D/trace.out 2>&1
  step env-check python3 $TOOLS/gate_check.py env $D > $D/env-check.txt 2>&1
}
