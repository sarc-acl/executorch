#!/bin/bash
# chain2.sh [job to wait for]: device side. After the controls (chain1), with build topic1 (hook + candidate 1 code):
#   s1-aa      baseline and A/A, record-only clock: the parent build against topic1, both with the parent
#              environment (the first campaign's final stack). The A/A also shows what the hook and the linked dev
#              code cost when the fused node is not selected. Then its content check and warm traces of both arms.
#   s2n-noenv  unmodified verify.sh on topic1 with nothing selected against the parent's s0n-noenv (owner decision
#              D4: nothing selected, nothing changed).
cd "$(dirname "$0")"; source ./common.sh
[[ -n ${1:-} ]] && while ! grep -q "^DONE\|^KILLED" $A/jobs/$1.status; do sleep 20; done
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; echo "== rc=$rc $1"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && { echo "CHAIN_STOPPED rc=$rc"; exit $rc; }; return 0; }
PENV="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5 ET_VK_SARC_SOFTMAX_VARIANT=orin_g64"
D=$A/stage/s1-aa
step ./stage.sh s1-aa parent "$PENV" topic1 "$PENV" "baseline + A/A: the parent build against topic1 (hook + candidate 1 code), both with the parent environment"
step ./session.sh s1-aa --calibrate > $D/e2e5.out 2>&1; tail -3 $D/e2e5.out
python3 summarize.py $D/raw > $D/raw/summary.csv 2>&1; cat $D/raw/summary.csv
python3 gate_check.py session $D/raw --calibration --require-logs > $D/session-check.txt 2>&1; tail -2 $D/session-check.txt
step ./trace.sh s1-aa 1b,3b,8b 4w,8da4w "parent cand" > $D/trace.out 2>&1; tail -2 $D/trace.out
step env PARENT_CTL_NAME=s0n-noenv ./noenv_verify.sh topic1 s2n-noenv
echo CHAIN_DONE
