#!/bin/bash
# hook_control.sh <session>: the "nothing selected" control of the owner decision 2026-10-05 (release-zone hooks).
# <session> is staged (stage.sh) with an EMPTY candidate environment on the build that carries the hook. The
# unmodified sarc/tools/verify.sh runs on it once and is compared with the parent control (s0-parent-verify):
#   - item by item, as in every gate (gate_check.py verify);
#   - line by line: control.diff is the diff of the two verify.out files with the measured rates
#     (prefill_tok_s, decode_tok_s) replaced by "<rate>"; every other character, the dispatched kernel lists
#     included, must be equal.
# control.done says CONTROL_SAME, CONTROL_DIFFERS or the step that failed. No timed session and no trace: this is
# not a candidate gate and writes no gate.done.
source "$(dirname "$0")/common.sh"; source $TOOLS/gatelib.sh; S=$1; D=$A/stage/$S
[[ -e $D/control.done ]] && { echo "session $S already has a control: $(cat $D/control.done)" >&2; exit 2; }
need $D/STAGE.md $PARENT_CTL/verify.out; cand_env > /dev/null
[[ -z $ENVS ]] || { echo "the control needs an empty candidate environment, staged is [$ENVS]" >&2; exit 77; }
grep -q GATE_ACCEPTED $PARENT_CTL/gate.done || { echo "no accepted parent control" >&2; exit 77; }
finish() { echo "$1 $(date -u +%FT%TZ) $2" | tee $D/control.done; exit $3; }
cool_start 50 300
step verify run_verify ""
step verify-check python3 $TOOLS/gate_check.py verify $D $PARENT_CTL > $D/verify-check.txt 2>&1
norm() { sed -E 's/(prefill_tok_s|decode_tok_s)=[0-9.]+/\1=<rate>/g' "$1"; }
if diff <(norm $PARENT_CTL/verify.out) <(norm $D/verify.out) > $D/control.diff; then
  finish CONTROL_SAME "verify.out equals the parent control line by line ($(wc -l < $D/verify.out) lines, rates aside); verify-check ACCEPT" 0
fi
finish CONTROL_DIFFERS "see control.diff" 1
