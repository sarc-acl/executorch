#!/bin/bash
# gate.sh <session> "<cand env>" [--sdpa]: the full gate for one staged candidate, one GPU job at a time.
#   0. (--sdpa, for a candidate that changes an SDPA kernel) 12 passes each of --sdpa-correctness-only with
#      --sdpa-tier=all, extended and full under the candidate env, and the SDPA perf suite with and without it;
#   1. sarc/tools/verify.sh (unmodified) on the staged candidate binaries with the candidate env:
#      correctness, dispatched kernels, production-diff 1B/3B/8B x buffer/texture3d (nonzero zp for 8da4w),
#      e2e tiled vs default, next token on the real-text and the unaligned prompt, decode;
#   2. e2e5.sh parent vs candidate, 5 valid runs per cell, with the calibrated clock threshold;
#   3. warm ETDump traces of both arms (1b,3b,8b);
#   4. gate_check.py over all of it, including the comparison with the parent control (parent_verify.sh).
# Every step runs even after an earlier failure, so a failed candidate leaves complete evidence; nothing is
# deleted. gate.done holds GATE_PASS or GATE_FAIL and gate.txt the per-requirement lines; exit status follows.
# A foreign GPU process aborts the gate (GATE_ABORTED, exit 76): stop measuring and report it.
. "$(dirname "$(readlink -f "$0")")/host.sh"; S=$1; ENVS=$2; SDPA=${3:-}; D=$A/stage/$S
PV=$A/stage/s0-parent-verify/verify.out
[[ -x $D/test_llama_microbench && -f $D/STAGE.md ]] || { echo "session $S is not staged" >&2; exit 2; }
[[ -e $D/gate.done ]] && { echo "session $S was already gated; stage a new session" >&2; exit 2; }
[[ -s $CLKMIN_FILE && -s $IDLE_FILE ]] || { echo "no calibration: run the baseline session with --calibrate first" >&2; exit 2; }
grep -q 'VERIFY_DONE rc=0' $PV 2>/dev/null || { echo "no parent control: run parent_verify.sh first" >&2; exit 2; }
aborted() { echo "GATE_ABORTED $1" | tee $D/gate.done; exit 76; }
if [[ $SDPA == --sdpa ]]; then
  $TOOLS/sdpa_passes.sh $S cand 12 "$ENVS" > $D/sdpa.out 2>&1; [[ $? == 76 ]] && aborted "foreign GPU process during the SDPA passes"
  O=$D/sdpa-correctness
  env $ENVS $TOOLS/gl.sh $D/test_llama_microbench --sdpa --json-out=$O/perf-cand.json > $O/perf-cand.log 2>&1; echo "perf cand rc=$?" >> $D/sdpa.out
  $TOOLS/gl.sh $D/test_llama_microbench --sdpa --json-out=$O/perf-table.json > $O/perf-table.log 2>&1; echo "perf table rc=$?" >> $D/sdpa.out
fi
cool_start
guarded $D/verify.others env $ENVS $ET/sarc/tools/verify.sh --dir $D --lock $LOCK \
  --models 1b,3b,8b --schemes 4w,8da4w --pdiff --out verify > $D/verify.out 2>&1
rc=$?; echo "VERIFY_DONE rc=$rc" >> $D/verify.out; [[ $rc == 76 ]] && aborted "foreign GPU process during verify.sh"
$TOOLS/session.sh $S > $D/e2e5.out 2>&1; [[ $? == 76 ]] && aborted "foreign GPU process during the timing session"
$TOOLS/trace.sh $S 1b,3b,8b 4w,8da4w "parent cand" > $D/trace.out 2>&1; [[ $? == 76 ]] && aborted "foreign GPU process during the traces"
python3 $TOOLS/summarize.py $D/raw > $D/raw/summary.csv 2>&1
python3 $TOOLS/gate_check.py $D --parent $PV $SDPA > $D/gate.txt 2>&1; rc=$?
tail -1 $D/gate.txt > $D/gate.done; cat $D/gate.txt; exit $rc
