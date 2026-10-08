#!/bin/bash
# q-c3.sh (GPU host, via rstart): candidate 3 (linear kernel per layer shape, profile 7900xtx-refine2, build c3) against candidate 1
# (parent binary, ET_VK_SARC_780M_PROFILE=c7): gate (timed session, verify.sh against the snapshot, traces; no SDPA tiers: no attention
# kernel changes) and the byte comparison of every prefill linear output; then candidate 4 (the texel-wise 8da4w family, profile
# 7900xtx-refine3) against candidate 3, the same binary, the same checks.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-c3.status; }
C3="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_7900XTX_PROFILE=7900xtx-refine2"; C4="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_7900XTX_PROFILE=7900xtx-refine3"
st "c3 gate"; $T/gate.sh c3-linear "$C3"; st "c3 gate done"
$T/linear_bitwise.sh c3-linear > $A/stage/c3-linear/linear-bitwise.out 2>&1; st "c3 linear bitwise: $(tail -1 $A/stage/c3-linear/linear-bitwise.out)"
st "c4 gate"; $T/gate.sh c4-texel "$C4"; st "c4 gate done"
$T/linear_bitwise.sh c4-texel > $A/stage/c4-texel/linear-bitwise.out 2>&1; st "c4 linear bitwise: $(tail -1 $A/stage/c4-texel/linear-bitwise.out)"
st Q_C3_DONE
