#!/bin/bash
# q-final2.sh (GPU host, via rstart): the final verification (R11) redone on the build c9 of the committed head's code (commit a554a5791, with the second-set variants, all default off):
# (1) nothing selected = nothing changed: unmodified verify.sh on the build with the parent's environment only, compared with s0-parent-verify; (2) the final stack (profile 7900xtx-refine5)
# against the pristine parent: gate (one timed session, SDPA tiers 12 passes each with 1 control pass, verify.sh against s0, traces), SDPA output / reference-error evidence, real-text probe.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-final2.status; }
F="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_7900XTX_PROFILE=7900xtx-refine5"
st "inert verify"; $T/verify_stage.sh inert "ET_VK_SARC_UNVERIFIED=1"
/usr/bin/python3 $T/verify_compare.py $A/stage/s0-parent-verify/verify.out $A/stage/inert/verify.out > $A/stage/inert/verify-compare.txt 2>&1; st "inert verify done: $(tail -1 $A/stage/inert/verify-compare.txt)"
st "final2 gate, env [$F]"; $T/gate.sh final2 "$F" sdpa; st "final2 gate done"
$T/sdpa_evidence.sh final2 "all extended peaked full" > $A/stage/final2/sdpa-evidence.out 2>&1; st "final2 sdpa evidence done: $(head -1 $A/stage/final2/sdpa-evidence.out)"
$T/probe_run.sh final2 > $A/stage/final2/probe.out 2>&1; st "final2 probe done: $(tail -1 $A/stage/final2/probe.out)"
st Q_FINAL2_DONE
