#!/bin/bash
# sdpa_evidence.sh <session> [tiers="all extended peaked full"]: SDPA output evidence of stage/<session>, parent env
# against candidate env (stage/<session>/{parent,cand}/env), on the same inputs (a fixed function of each case):
#   - every case's raw fp16 output (ET_VK_DUMP_OUTPUT_DIR), compared byte for byte (bit-identity claims, D3 last part)
#   - every case's rms / maximum error against the fp64 reference (ET_VK_SDPA_ERROR_REPORT=1; reference-error rule D3.1)
# One gl.sh job per arm and tier. Output stage/<session>/sdpa-error/{<arm>-<tier>.log,dump-<arm>-<tier>/,bitwise.txt,error.csv}
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$A/stage/$1; TIERS=${2:-all extended peaked full}; O=$S/sdpa-error; mkdir -p $O
for tier in $TIERS; do for arm in parent cand; do
  e=(); mapfile -t e < $S/$arm/env; mkdir -p $O/dump-$arm-$tier
  env "${e[@]}" ET_VK_SDPA_ERROR_REPORT=1 ET_VK_DUMP_OUTPUT_DIR=$O/dump-$arm-$tier $T/gl.sh $S/test_llama_microbench \
    --sdpa-correctness-only --sdpa-tier=$tier > $O/$arm-$tier.log 2>&1
  echo "$arm $tier rc=$? passed=$(grep -c PASSED $O/$arm-$tier.log) failed=$(grep -c FAILED $O/$arm-$tier.log)" >> $O/runs.txt
done
  for f in $O/dump-parent-$tier/*.bin; do b=$(basename $f)
    if cmp -s $f $O/dump-cand-$tier/$b; then echo "$tier $b IDENTICAL"; else echo "$tier $b DIFFERS"; fi
  done >> $O/bitwise.txt
done
/usr/bin/python3 $T/sdpa_error_table.py $O $O/error.csv > $O/error.out 2>&1
echo "bitwise: $(grep -c IDENTICAL $O/bitwise.txt) identical, $(grep -c DIFFERS $O/bitwise.txt) differ"; tail -1 $O/error.out
