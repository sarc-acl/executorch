#!/bin/bash
# q-phases.sh (GPU host, via rstart): phase timing (shader clock) of the 8da4w table kernel t128x64k32g42s32 through its PROF twin
# (sarc_dev_prof_dq8ca_zpg_t128x64k32g42s32p) on the twelve prefill shapes, on the parent build's microbench (stage c1-softmax), after q-screens2.
# The 4w table kernel (t256x128k32g24s32f32cbt) has no twin in the dev zone; a twin would be new shader work and is not attempted (see STATUS.md).
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-phases.status; }
st "waiting for q-screens2"; until grep -q Q_SCREENS2_DONE $A/logs/q-screens2.status 2>/dev/null; do sleep 30; done
st "phases 8da4w table kernel"; $T/phases.sh c1-softmax parent-8da4w 8da4w sarc_dev_prof_dq8ca_zpg_t128x64k32g42s32p 128 64
st Q_PHASES_DONE
