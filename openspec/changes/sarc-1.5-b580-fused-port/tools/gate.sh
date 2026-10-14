#!/bin/bash
# gate.sh <session> [--sdpa]: the full gate for one staged candidate, one GPU job at a time.
# The candidate environment is the staged file stage/<session>/cand/env and nothing else: the SDPA passes,
# verify.sh, the timing session and the traces all read that one file, and the binaries verify.sh runs (top
# level of the session) must be byte-identical to cand/, so correctness is gated on exactly the configuration
# that is timed. There is no way to pass an environment on the command line.
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
. "$(dirname "$(readlink -f "$0")")/host.sh"; S=${1:?session}; SDPA=${2:-}; D=$A/stage/$S
gpu_shared || exit 75
[[ $# -le 2 && ( -z $SDPA || $SDPA == --sdpa ) ]] || { echo "usage: gate.sh <session> [--sdpa] (the environment comes from cand/env)" >&2; exit 2; }
PV=$A/stage/s0-parent-verify/verify.out
[[ -x $D/test_llama_microbench && -f $D/STAGE.md ]] || { echo "session $S is not staged" >&2; exit 2; }
[[ -f $D/cand/env ]] && cmp -s $D/cand/env $D/cand-traced/env || { echo "cand/env missing or different from cand-traced/env" >&2; exit 2; }
for f in llama_main libllama_runner.so; do cmp -s $D/$f $D/cand/$f || { echo "$D/$f is not the staged candidate binary" >&2; exit 2; }; done
grep -qvE '^[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*$' $D/cand/env && { echo "cand/env must hold one KEY=VALUE per line, no spaces" >&2; exit 2; }
mapfile -t CENV < $D/cand/env; ENVS="${CENV[*]}"
[[ -e $D/gate.done ]] && { echo "session $S was already gated; stage a new session" >&2; exit 2; }
[[ -s $CLKMIN_FILE && -s $IDLE_FILE && -s $BUSYMAX_FILE ]] || { echo "no calibration: run the baseline session with --calibrate first" >&2; exit 2; }
grep -q 'VERIFY_DONE rc=0' $PV 2>/dev/null || { echo "no parent control: run parent_verify.sh first" >&2; exit 2; }
{ echo "gate $(date -u +%FT%TZ) sdpa=[$SDPA] candidate env (cand/env, sha256 $(sha256sum < $D/cand/env | cut -c1-16)): [$ENVS]"; } > $D/gate.env
aborted() { echo "GATE_ABORTED $1" | tee $D/gate.done; exit 76; }
if [[ $SDPA == --sdpa ]]; then
  $TOOLS/sdpa_passes.sh $S cand 12 "$ENVS" > $D/sdpa.out 2>&1; [[ $? == 76 ]] && aborted "foreign GPU process during the SDPA passes"
  O=$D/sdpa-correctness
  env $ENVS $TOOLS/gl.sh $D/test_llama_microbench --sdpa --json-out=$O/perf-cand.json > $O/perf-cand.log 2>&1
  rc=$?; echo "perf cand rc=$rc" >> $D/sdpa.out; [[ $rc == 76 ]] && aborted "foreign GPU process during the SDPA perf suite (candidate)"
  $TOOLS/gl.sh $D/test_llama_microbench --sdpa --json-out=$O/perf-table.json > $O/perf-table.log 2>&1
  rc=$?; echo "perf table rc=$rc" >> $D/sdpa.out; [[ $rc == 76 ]] && aborted "foreign GPU process during the SDPA perf suite (table)"
fi
cool_start
guarded $D/verify.others env $ENVS $ET/sarc/tools/verify.sh --dir $D --lock $LOCK \
  --models 1b,3b,8b --schemes 4w,8da4w --pdiff --out verify > $D/verify.out 2>&1
rc=$?; echo "VERIFY_DONE rc=$rc" >> $D/verify.out; [[ $rc == 76 ]] && aborted "foreign GPU process during verify.sh"
$TOOLS/session.sh $S --reps ${B580_REPS:-5} > $D/e2e5.out 2>&1; [[ $? == 76 ]] && aborted "foreign GPU process during the timing session"
$TOOLS/trace.sh $S 1b,3b,8b 4w,8da4w "parent cand" > $D/trace.out 2>&1; [[ $? == 76 ]] && aborted "foreign GPU process during the traces"
python3 $TOOLS/summarize.py $D/raw > $D/raw/summary.csv 2>&1
python3 $TOOLS/gate_check.py $D --parent $PV $SDPA > $D/gate.txt 2>&1; rc=$?
tail -1 $D/gate.txt > $D/gate.done; cat $D/gate.txt; exit $rc
