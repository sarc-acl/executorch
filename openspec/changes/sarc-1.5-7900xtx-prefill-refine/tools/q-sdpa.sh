#!/bin/bash
# q-sdpa.sh (GPU host, via rstart): after q-c3: the kernel-level screen (3 rounds) of the unfused QK^T / attn*V kernels that the dev zone names
# (ET_VK_SARC_DEV_PROFILE profiles: QK^T without the mask fill, packed K staging, attn*V multi-load tiles), on the parent build's microbench
# of stage c1-softmax, base environment = candidate 1 (softmax r3). Kernel timings, not a timed session.
source "$(dirname "$(readlink -f "$0")")/env.sh"
st() { echo "$(date -u +%FT%TZ) $*" >> $A/logs/q-sdpa.status; }
st "waiting for q-c3"; until grep -q Q_C3_DONE $A/logs/q-c3.status 2>/dev/null; do sleep 30; done
D=$A/stage/c1-softmax; echo "ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7" > $D/sdpa-screen.env
PS=(table qk-t128x64k32g22s64nf qk-t128x64k32g42s32nf qk-t128x64k32g42s32 qk-t128x64k32g24s32nf qk-t128x64k32g22s32nf
  qkpk-t128x64k32g22s64nf qkpk-t128x64k64g22s64nf qkpk-t128x64k32g42s32nf qkpk-t128x64k64g42s32nf qkpk-t128x128k32g42s32nf qkpk-t64x64k32g22s32nf
  qkpk-t128x64k32g24s32nf qkpk-t256x64k32g42s32nf av-t64x64k32g42s32 av-t64x64k32g24s32 avml-t64x128k32g42s32 avml-t64x128k32g22s64
  avml-t128x128k32g42s32 avml-t128x64k32g42s32)
st "sdpa screen (${#PS[@]} profiles)"; $T/sdpa_screen.sh c1-softmax 3 $D/sdpa-screen.csv $D/sdpa-screen.env "${PS[@]}" > $A/logs/sdpa-screen.out 2>&1
st "sdpa screen done"; st Q_SDPA_DONE
