#!/bin/bash
# sdpa_ref.sh <name> <build tag> <profile>: error of the SDPA kernels against the fp32 CPU reference, parent arm
# and candidate arm on the same inputs (owner decision 2026-10-04, second, item 1). Both arms run the SAME test
# binary (build/<tag>/tests/test_llama_microbench --sdpa-correctness-only):
#   parent    the parent environment of this campaign (host.sh PARENT_ENV: xe2-refine5, the three attention kernels)
#   <profile> ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=<profile>
# Tiers full and extended decide the exit status (every case: rms and maximum error not larger than the
# parent's); tier peaked (sharp attention rows, added with the fused kernel) is run and tabulated the same way
# but only reported. Output: raw/<name>/{parent,<profile>}-<tier>.log and raw/<name>/<tier>.csv (tools/sdpa_error.py).
. "$(dirname "$(readlink -f "$0")")/host.sh"; O=$A/raw/${1:?name}; B=$A/build/${2:?build}/tests/test_llama_microbench; P=${3:?profile}
mkdir -p $O || exit 2; { sha256sum $B; date -u; echo "parent env: $PARENT_ENV"; } > $O/env.txt; st=0
for tier in full extended peaked; do
  cool_start; env $PARENT_ENV $TOOLS/gl.sh $B --sdpa-correctness-only --sdpa-tier=$tier > $O/parent-$tier.log 2>&1; rc=$?; echo "parent $tier rc=$rc"; [[ $rc == 75 || $rc == 76 ]] && exit $rc
  cool_start; env ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=$P $TOOLS/gl.sh $B --sdpa-correctness-only --sdpa-tier=$tier > $O/$P-$tier.log 2>&1; rc=$?; echo "$P $tier rc=$rc"; [[ $rc == 75 || $rc == 76 ]] && exit $rc
  python3 $TOOLS/sdpa_error.py parent=$O/parent-$tier.log $P=$O/$P-$tier.log > $O/$tier.csv; r=$?; echo "$tier: $(tail -1 $O/$tier.csv)"; [[ $r == 0 || $tier == peaked ]] || st=1
done
exit $st
