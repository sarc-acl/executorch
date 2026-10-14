#!/bin/bash
# timed.sh <session> ["<cand env>"]: device side. The timed part of gate.sh alone, for a candidate whose full gate
# already ran in another session with the same binaries and environment: e2e5.sh parent vs candidate (six cells,
# 5 valid runs per arm, next token on the four prompts), its content check, the warm ETDump traces of both arms
# and the environment check. No verify.sh here; stage/<session>/gate.done says SESSION_ACCEPTED, or GATE_REJECTED /
# GATE_ABORTED with the step, as gate.sh does.
source "$(dirname "$0")/common.sh"; source $TOOLS/gatelib.sh; S=$1; D=$A/stage/$S
[[ -e $D/gate.done ]] && { echo "session $S already done: $(cat $D/gate.done)" >&2; exit 2; }
near_tie_arg; need $D/STAGE.md $CLKFILE; cand_env "${@:2}"
step session $TOOLS/session.sh $S --clkmin-file $CLKFILE > $D/e2e5.out 2>&1
python3 $TOOLS/summarize.py $D/raw > $D/raw/summary.csv 2>&1
step session-check python3 $TOOLS/gate_check.py session $D/raw --clkmin $CLKFILE --require-logs ${NT:-} > $D/session-check.txt 2>&1
step trace $TOOLS/trace.sh $S 1b,3b,8b 4w,8da4w "parent cand" > $D/trace.out 2>&1
step env-check python3 $TOOLS/gate_check.py env $D --timed-only > $D/env-check.txt 2>&1
finish SESSION_ACCEPTED "timed session, traces and environment check passed (no verify.sh in this session); the gain is in raw/summary.csv" 0
