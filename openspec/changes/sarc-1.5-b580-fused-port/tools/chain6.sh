#!/bin/bash
# chain6.sh <commit>: build topic5 (six more multi-subgroup variants: g4 / g8, wider blocks); per new variant one
# correctness pass of tiers extended, peaked and fused (not timed) and the compiler's statistics of its pipeline
# (INTEL_DEBUG=cs: instructions, spills:fills; a compile-time dump of this process only); then a one-round look
# screen4-fused (idle wait and cooling per run) beside the parent's kernels and the best variants so far.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"; ST=$A/logs/chain6.status; REV=${1:?commit}
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
say "chain6 start $REV"
$TOOLS/build-both.sh topic5 $REV > $A/logs/build-topic5.out 2>&1 || { say "CHAIN5_STOPPED topic5 build failed"; exit 1; }
say "topic5 built"
V="d128_t16x128s16m8g8oj d128_t32x64s16m8g8oj d128_t8x128s16m8g8roj d64_t16x64s16m8g4oj d64_t16x128s16m8g4roj d64_t8x64s16m8g4roj"
O=$A/raw/c1-smoke4; mkdir -p $O; T=$A/build/topic5/tests/test_llama_microbench
for v in $V; do line="smoke $v:"
  for tier in extended peaked fused; do
    env ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-fused-$v $TOOLS/gl.sh $T --sdpa-correctness-only --sdpa-tier=$tier > $O/$v-$tier.log 2>&1
    line+=" $tier rc=$? ok=$(grep -c 'mismatches=0/.*PASSED' $O/$v-$tier.log)/$(grep -c 'mismatches=' $O/$v-$tier.log) fused=$(grep -c 'fused=sarc_dev' $O/$v-$tier.log);"
  done
  env MESA_SHADER_CACHE_DISABLE=true INTEL_DEBUG=cs ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-fused-$v $TOOLS/gl.sh $T --sdpa-correctness-only --sdpa-tier=fused 2>&1 | grep -A1 '^Native code for' | grep 'SIMD' | sort | uniq -c > $O/$v-compile.txt
  say "$line compile: $(grep -v 'SIMD32 shader: 535 ' $O/$v-compile.txt | sed 's/ sends.*//; s/^ *[0-9]* //' | tr '\n' '|')"
done
P="b580-refine3 b580-fused-d64_t16x64s16m8g4roj b580-fused-d64_t8x32s16m8g2roj b580-fused-d128_t16x64s16m8g4oj b580-fused-d128_t8x64s16m8g4roj"; for v in $V; do P+=" b580-fused-$v"; done
$TOOLS/screen_sdpa.sh screen4-fused topic5 1 $P > $A/logs/screen4-fused.out 2>&1; rc=$?
python3 $TOOLS/screen_sdpa_summary.py $A/raw/screen4-fused/screen.csv b580-refine3 > $A/raw/screen4-fused/summary.csv 2>&1
say "screen4-fused rc=$rc $(tail -1 $A/raw/screen4-fused/env.txt)"
say "CHAIN5_DONE"
