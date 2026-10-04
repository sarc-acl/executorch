#!/bin/bash
# gatelib.sh: shared steps of gate.sh, gate_sdpa.sh and parent_verify.sh (sourced after common.sh, with D = the
# stage directory). A failed step ends the gate: no further GPU job runs and gate.done records the step.
# Device loss (exit 70) is passed up unchanged.
PARENT_CTL=$A/stage/s0-parent-verify
finish() { echo "$1 $(date -u +%FT%TZ) $2" | tee $D/gate.done; exit $3; }
step() { # step <name> <command...>
  local name=$1; shift; "$@"; local rc=$?
  [[ $rc == 70 || -f $GONE ]] && finish GATE_ABORTED "device lost during $name" 70
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
  step session $TOOLS/session.sh $(basename $D) > $D/e2e5.out 2>&1
  python3 $TOOLS/summarize.py $D/raw > $D/raw/summary.csv 2>&1
  step session-check python3 $TOOLS/gate_check.py session $D/raw > $D/session-check.txt 2>&1
  step trace $TOOLS/trace.sh $(basename $D) 1b,3b,8b 4w,8da4w "parent cand" > $D/trace.out 2>&1
}
