#!/bin/bash
# q-c7.sh (GPU host, via rstart): candidate 7 (profile 7900xtx-refine6 = refine5 plus the sweep 128 x 64 tile on 8B w2; build c10) against candidate 6 / refine5 (the same build):
# gate (timed session, verify.sh against the snapshot, traces; no SDPA tiers: no attention kernel changes) and the byte comparison of every prefill linear output.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-c7.status; }
C7="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_7900XTX_PROFILE=7900xtx-refine6"
st "c7 gate"; $T/gate.sh c7-w2 "$C7"; st "c7 gate done"
$T/linear_bitwise.sh c7-w2 > $A/stage/c7-w2/linear-bitwise.out 2>&1; st "c7 linear bitwise: $(tail -1 $A/stage/c7-w2/linear-bitwise.out)"
st Q_C7_DONE
