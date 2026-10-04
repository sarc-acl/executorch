#!/bin/bash
# gate.sh <session> "<cand env>": the full gate for one candidate, one GPU job at a time:
#   1. sarc/tools/verify.sh (unmodified) on the staged candidate binaries with the candidate env:
#      correctness, dispatched kernels, production-diff 1B/3B/8B x buffer/texture3d (nonzero zp for 8da4w),
#      e2e tiled vs default, next token on the real-text and the unaligned prompt, decode
#   2. e2e5.sh parent vs candidate, 5 valid runs per cell
#   3. warm ETDump traces of both arms (1b,3b,8b)
. "$(dirname "$(readlink -f "$0")")/host.sh"; S=$1; ENVS=$2
cool_start
env $ENVS $ET/sarc/tools/verify.sh --dir $A/stage/$S --lock $LOCK \
  --models 1b,3b,8b --schemes 4w,8da4w --pdiff --out verify > $A/stage/$S/verify.out 2>&1
echo "VERIFY_DONE rc=$?" >> $A/stage/$S/verify.out
$TOOLS/session.sh $S > $A/stage/$S/e2e5.out 2>&1
$TOOLS/trace.sh $S 1b,3b,8b 4w,8da4w "parent cand" > $A/stage/$S/trace.out 2>&1
python3 $TOOLS/summarize.py $A/stage/$S/raw > $A/stage/$S/raw/summary.csv 2>&1
echo GATE_DONE > $A/stage/$S/gate.done
