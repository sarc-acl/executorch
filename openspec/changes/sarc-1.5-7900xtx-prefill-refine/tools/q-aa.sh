#!/bin/bash
# q-aa.sh (GPU host, via rstart): stage `aa` = the parent build in both arms: A/A session, 7 repeats (its parent arm is the baseline
# check); then the snapshot s0-parent-verify (unmodified verify.sh on the parent build) and the SDPA tiers of the parent, 1 pass each.
# Stages are pushed first (push-stage.sh aa; push-stage.sh s0-parent-verify). Status lines in logs/q-aa.status.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-aa.status; }
st "start (aa)"
$T/e2e5.sh --stage $A/stage/aa --out raw --reps 7 --extra 3 > $A/stage/aa/e2e5.out 2>&1
/usr/bin/python3 $T/summarize.py $A/stage/aa/raw 5 > $A/stage/aa/raw/summary.csv 2>&1
/usr/bin/python3 $T/summarize.py $A/stage/aa/raw 7 > $A/stage/aa/raw/summary7.csv 2>&1
st "aa done"
$T/verify_stage.sh s0-parent-verify "ET_VK_SARC_UNVERIFIED=1"
st "s0 verify done"
$T/sdpa_tiers.sh s0-parent-verify "ET_VK_SARC_UNVERIFIED=1" 1 "all extended full" table
st "s0 sdpa tiers done"; st Q_AA_DONE
