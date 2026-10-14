#!/bin/bash
# sdpa_ref.sh <name> <build tag> <profile>: error of the SDPA kernels against the fp32 CPU reference, parent arm
# and candidate arm on the same inputs (owner decision 2026-10-04, second, item 1). Both arms run the SAME test
# binary (build/<tag>/tests/test_llama_microbench --sdpa-correctness-only, tiers full and extended):
#   parent   no environment: the stock kernels the parent runs on this card (no SDPA row in the release tables)
#   <profile> ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=<profile>
# The parent build's own test binary does not print the [sdpa-error] lines (they were added in the dev zone by
# the B70 campaign), which is why the topic binary runs both arms. Output: raw/<name>/{parent,<profile>}-<tier>.log
# and raw/<name>/<tier>.csv (tools/sdpa_error.py: side by side, with the not-larger-than-the-parent columns).
. "$(dirname "$(readlink -f "$0")")/host.sh"; O=$A/raw/${1:?name}; B=$A/build/${2:?build}/tests/test_llama_microbench; P=${3:?profile}
mkdir -p $O || exit 2; { sha256sum $B; date -u; } > $O/env.txt; st=0
for tier in full extended; do
  cool_start; $TOOLS/gl.sh $B --sdpa-correctness-only --sdpa-tier=$tier > $O/parent-$tier.log 2>&1; rc=$?; echo "parent $tier rc=$rc"; [[ $rc == 75 || $rc == 76 ]] && exit $rc
  cool_start; env ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=$P $TOOLS/gl.sh $B --sdpa-correctness-only --sdpa-tier=$tier > $O/$P-$tier.log 2>&1; rc=$?; echo "$P $tier rc=$rc"; [[ $rc == 75 || $rc == 76 ]] && exit $rc
  python3 $TOOLS/sdpa_error.py parent=$O/parent-$tier.log $P=$O/$P-$tier.log > $O/$tier.csv; r=$?; tail -1 $O/$tier.csv; [[ $r == 0 ]] || st=1
done
exit $st
