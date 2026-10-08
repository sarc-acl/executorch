#!/bin/bash
# gate_rest.sh <session>: the remaining steps of a gate that ended GATE_REJECTED at verify-check, run for evidence
# only: the six-cell session with its content check, the warm traces and the environment check. It changes
# nothing about the verdict: gate.done stays as written by the gate, and the outcome of these steps goes to
# evidence.done ("EVIDENCE_COMPLETE" or the failed step). Refuses any session that was not rejected at
# verify-check (an aborted or accepted session has nothing to add here).
source "$(dirname "$0")/common.sh"; source $TOOLS/gatelib.sh; S=$1; D=$A/stage/$S
grep -q '^GATE_REJECTED .* step verify-check failed' $D/gate.done 2>/dev/null || { echo "session $S was not rejected at verify-check" >&2; exit 2; }
[[ -e $D/evidence.done ]] && { echo "already run: $(cat $D/evidence.done)" >&2; exit 2; }
need $D/STAGE.md $CLKFILE; cand_env > /dev/null
finish() { echo "$1 $(date -u +%FT%TZ) $2 (gate verdict unchanged: $(cut -d' ' -f1 $D/gate.done))" | tee $D/evidence.done; exit $3; }
cool_start 50 300
timed_and_traced
finish EVIDENCE_COMPLETE "session, session-check, trace and env-check passed" 0
