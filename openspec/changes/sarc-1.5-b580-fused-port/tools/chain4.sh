#!/bin/bash
# chain4.sh <commit>: build topic3 (register-saving variants j / a of the fused kernel), one correctness pass of
# tiers all, extended, peaked and fused per new variant (not timed), then a one-round look screen2-fused
# (idle wait and cooling per run) of the new variants beside the parent's kernels and the best two of screen 1.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; ST=$A/logs/chain4.status; REV=${1:?commit}
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
say "chain4 start $REV"
$TOOLS/build-both.sh topic3 $REV > $A/logs/build-topic3.out 2>&1 || { say "CHAIN4_STOPPED topic3 build failed"; exit 1; }
say "topic3 built"
V="d64_t8x32s16m8roj d64_t16x32s16m8roj d64_t8x32s16m8roja d64_t16x32s16m8roja d64_t16x64s16m8roja d64_t16x32s16m8oa
   d128_t8x64s16m8roj d128_t8x64s16m8roja d128_t16x64s16m8roja d128_t8x64s16m8oa d128_t16x64s16m8oa d128_t8x32s16m8roja d128_t16x32s16m8roja d128_t8x64s16m8rja"
O=$A/raw/c1-smoke2; mkdir -p $O; T=$A/build/topic3/tests/test_llama_microbench
for v in $V; do line="smoke $v:"
  for tier in all extended peaked fused; do
    env ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-fused-$v $TOOLS/gl.sh $T --sdpa-correctness-only --sdpa-tier=$tier > $O/$v-$tier.log 2>&1
    line+=" $tier rc=$? ok=$(grep -c 'mismatches=0/.*PASSED' $O/$v-$tier.log)/$(grep -c 'mismatches=' $O/$v-$tier.log) fused=$(grep -c 'fused=sarc_dev' $O/$v-$tier.log);"
  done; say "$line"
done
P="b580-refine3 b580-fused-d64_t8x32s16m8ro b580-fused-d128_t8x64s16m8ro"; for v in $V; do P+=" b580-fused-$v"; done
$TOOLS/screen_sdpa.sh screen2-fused topic3 1 $P > $A/logs/screen2-fused.out 2>&1; rc=$?
python3 $TOOLS/screen_sdpa_summary.py $A/raw/screen2-fused/screen.csv b580-refine3 > $A/raw/screen2-fused/summary.csv 2>&1
say "screen2-fused rc=$rc $(tail -1 $A/raw/screen2-fused/env.txt)"
say "CHAIN4_DONE"
