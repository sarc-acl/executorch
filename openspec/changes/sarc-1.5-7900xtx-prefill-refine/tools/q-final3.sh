#!/bin/bash
# q-final3.sh (GPU host, via rstart): the final verification (R11) on the build c10 = the committed head's code (commit 5950764fa): (1) nothing selected = nothing changed (inert3), (2) the final stack (profile 7900xtx-refine5) against the pristine parent (final3): gate, SDPA evidence, probe.
# (1) nothing selected = nothing changed: unmodified verify.sh on the build with the parent's environment only, compared with s0-parent-verify; (2) the final stack (profile 7900xtx-refine5)
# against the pristine parent: gate (one timed session, SDPA tiers 12 passes each with 1 control pass, verify.sh against s0, traces), SDPA output / reference-error evidence, real-text probe.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-final3.status; }
F="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_7900XTX_PROFILE=7900xtx-refine5"
st "inert3 verify"; $T/verify_stage.sh inert3 "ET_VK_SARC_UNVERIFIED=1"
/usr/bin/python3 $T/verify_compare.py $A/stage/s0-parent-verify/verify.out $A/stage/inert3/verify.out > $A/stage/inert3/verify-compare.txt 2>&1; st "inert3 verify done: $(tail -1 $A/stage/inert3/verify-compare.txt)"
st "final3 gate, env [$F]"; $T/gate.sh final3 "$F" sdpa; st "final3 gate done"
$T/sdpa_evidence.sh final3 "all extended peaked full" > $A/stage/final3/sdpa-evidence.out 2>&1; st "final3 sdpa evidence done: $(head -1 $A/stage/final3/sdpa-evidence.out)"
$T/probe_run.sh final3 > $A/stage/final3/probe.out 2>&1; st "final3 probe done: $(tail -1 $A/stage/final3/probe.out)"
st Q_FINAL2_DONE
