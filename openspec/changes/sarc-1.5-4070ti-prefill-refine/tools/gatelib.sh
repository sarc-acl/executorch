#!/bin/bash
# gatelib.sh: shared steps of gate.sh, gate_sdpa.sh and parent_verify.sh (sourced after common.sh, with D = the
# stage directory). A failed step ends the gate: no further GPU job runs and gate.done records the step.
# Device loss (exit 70), a busy lock (75) and a foreign GPU process (76) are passed up unchanged as GATE_ABORTED.
PARENT_CTL=$A/stage/s0-parent-verify
# The per-cell normal-clock thresholds (calibrate_clock.py on the baseline and A/A sessions). Candidate gates
# refuse to start without it and e2e5.sh applies it; a record-only session cannot pass gate_check.py.
CLKFILE=$CHANGE/results/4070ti/clkmin.json
# cand_env [given]: the ONE candidate environment of a session is the staged cand/env file (what e2e5.sh and
# trace.sh read). It must equal cand-traced/env; an environment given on the command line must be the same.
cand_env() {
  need $D/STAGE.md; [[ -f $D/cand/env && -f $D/cand-traced/env && -f $D/parent/env && -f $D/parent-traced/env ]] || { echo "session not staged with env files" >&2; exit 77; }
  cmp -s $D/cand/env $D/cand-traced/env && cmp -s $D/parent/env $D/parent-traced/env || { echo "env of the timed and traced binaries differ" >&2; exit 77; }
  ENVS=$(grep -v '^$' $D/cand/env | tr '\n' ' '); ENVS=${ENVS% }
  if [[ $# -gt 0 ]]; then local g; g=$(echo $1); [[ $g == "$ENVS" ]] || { echo "environment given ($g) is not the staged cand/env ($ENVS)" >&2; exit 77; }; fi
  echo "candidate environment: [$ENVS] sha256 $(sha256sum < $D/cand/env | cut -c1-16)" | tee $D/gate.env
}
finish() { echo "$1 $(date -u +%FT%TZ) $2" | tee $D/gate.done; exit $3; }
step() { # step <name> <command...>
  local name=$1; shift; "$@"; local rc=$?
  [[ $rc == 70 || -f $GONE ]] && finish GATE_ABORTED "device lost during $name" 70
  [[ $rc == 75 || $rc == 76 ]] && finish GATE_ABORTED "$name rc=$rc (75 lock busy, 76 foreign GPU process)" $rc
  gpu_alive || gpu_gone "after $name"
  [[ $rc == 0 ]] || finish GATE_REJECTED "step $name failed rc=$rc" $rc
}
run_verify() { # run_verify "<env>": the unmodified sarc/tools/verify.sh of the working copy
  need $D/llama_main $D/libllama_runner.so $D/test_llama_microbench $D/prompt_2048.txt $D/prompt_check.txt $D/r1304.txt
  no_others "before verify.sh"
  env $1 $ET/sarc/tools/verify.sh --dir $D --lock $LOCK --models 1b,3b,8b --schemes 4w,8da4w --pdiff --out verify > $D/verify.out 2>&1
  local rc=$?; echo "VERIFY_DONE rc=$rc" >> $D/verify.out; local o; o=$(others)
  [[ -n $o ]] && { echo "foreign GPU process after verify.sh: $o" >> $D/verify.out; return 76; }
  return $rc
}
timed_and_traced() { # the e2e session, its content check, then the warm traces
  step session $TOOLS/session.sh $(basename $D) --clkmin-file $CLKFILE > $D/e2e5.out 2>&1
  python3 $TOOLS/summarize.py $D/raw > $D/raw/summary.csv 2>&1
  step session-check python3 $TOOLS/gate_check.py session $D/raw --clkmin $CLKFILE --require-logs > $D/session-check.txt 2>&1
  step trace $TOOLS/trace.sh $(basename $D) 1b,3b,8b 4w,8da4w "parent cand" > $D/trace.out 2>&1
  step env-check python3 $TOOLS/gate_check.py env $D > $D/env-check.txt 2>&1
}
