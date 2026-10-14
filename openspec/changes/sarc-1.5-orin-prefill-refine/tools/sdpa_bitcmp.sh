#!/bin/bash
# sdpa_bitcmp.sh <out name> <build tag> <profile> <reference softmax variant> <softmax variant...>: device side.
# For a softmax variant that claims the arithmetic of the reference variant: the raw fp16 output of the attention
# block for every case of test_llama_microbench --sdpa-correctness-only, tiers extended and full (seeded inputs;
# ET_VK_SDPA_DUMP_DIR), compared byte for byte with the reference variant's. The reference is run twice (ref,
# ref2): its two dumps must be identical themselves, otherwise the comparison says nothing. Output:
# raw/<out name>/summary.txt: one line per (variant, case): identical, or the count of differing bytes; the
# softmax kernel that ran and the mismatch count of the test beside it.
T=$(dirname "$0"); source "$T/common.sh"; [[ $SIDE == device ]] || { echo "device only" >&2; exit 2; }
O=$A/raw/$1; BD=$A/build/$2/bundle; B=$BD/test_llama_microbench; P=$3; REF=$4; shift 4; need $B; mkdir -p $O
{ sha256sum $B; date -u; } >> $O/env.txt
run() { local d=$O/$1 v=$2 tier; [[ -s $d/full.log ]] && return 0; mkdir -p $d
  for tier in extended full; do cool_start 120
    env ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=$P ET_VK_SARC_SOFTMAX_VARIANT=$v ET_VK_SDPA_DUMP_DIR=$d LD_LIBRARY_PATH=$BD \
      $T/gl.sh $B --sdpa-correctness-only --sdpa-tier=$tier > $d/$tier.log 2>&1
    local rc=$?; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && exit $rc; done; return 0; }
run ref $REF; run ref2 $REF; for v in "$@"; do run $v $v; done
{ echo "variant,case,result,softmax_kernels,mismatch_lines"
  for v in ref2 "$@"; do
    k=$(grep -oh 'sarc_sdpa_attn_weights_softmax_buffer_half[a-z0-9_]*\|sdpa_attn_weights_softmax_[a-z0-9_]*' $O/$v/*.log | sort -u | tr '\n' ' ')
    m=$(grep -h '^\[sdpa-correctness\]' $O/$v/*.log | grep -vc 'mismatches=0/')
    for f in $O/ref/*.bin; do c=$(basename $f .bin)
      if [[ ! -s $O/$v/$c.bin ]]; then r=MISSING; elif cmp -s $f $O/$v/$c.bin; then r=identical; else r="DIFFER $(cmp -l $f $O/$v/$c.bin | wc -l) bytes of $(stat -c %s $f)"; fi
      echo "$v,$c,$r,$k,$m"; done; done; } > $O/summary.txt
echo "cases $(ls $O/ref/*.bin | wc -l); not identical: $(grep -vc ',identical,' $O/summary.txt | awk '{print $1 - 1}')"
