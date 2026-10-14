#!/bin/bash
# gate.sh <session> ["<cand env>"]: the full gate for one candidate that does not change an SDPA kernel, one GPU
# job at a time. Each step must pass on the CONTENT of its results (gate_check.py) before the next GPU job starts:
#   1. sarc/tools/verify.sh (unmodified) on the staged candidate binaries with the candidate env, compared item
#      by item with the parent control (parent_verify.sh): correctness, dispatched kernels, production-diff
#      1B/3B/8B x buffer/texture3d (nonzero zp for 8da4w), default vs tiled next token on prompt_check and the
#      unaligned prompt, decode;
#   2. e2e5.sh parent vs candidate: six cells with 5 valid runs per arm, next token SAME on both prompts;
#   3. warm ETDump traces of both arms (1b,3b,8b).
# stage/<session>/gate.done says GATE_ACCEPTED, GATE_REJECTED (with the step) or GATE_ABORTED (device lost).
# A rejected or aborted session is kept as it is; a new attempt takes a new session name.
# The candidate environment is the staged cand/env, used by every step; the optional argument must equal it.
# The calibrated clock file (results/orin/clkmin.json) must exist: the baseline and A/A sessions come first.
source "$(dirname "$0")/common.sh"; source $TOOLS/gatelib.sh; S=$1; D=$A/stage/$S
[[ -e $D/gate.done ]] && { echo "session $S already gated: $(cat $D/gate.done)" >&2; exit 2; }
near_tie_arg; need $D/STAGE.md $PARENT_CTL/verify.out $CLKFILE; cand_env "${@:2}"; grep -q GATE_ACCEPTED $PARENT_CTL/gate.done || { echo "no accepted parent control" >&2; exit 77; }
cool_start 300
step verify run_verify "$ENVS"
step verify-check python3 $TOOLS/gate_check.py verify $D $PARENT_CTL $NT > $D/verify-check.txt 2>&1
timed_and_traced
accepted
