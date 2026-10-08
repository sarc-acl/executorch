#!/bin/bash
# sdpa_err.sh <out name> <build tag> <arm>=<VAR=VALUE,...> [...]: device side. Error of the attention block
# against the fp32 CPU reference of test_llama_microbench (the [sdpa-error] line: maximum and rms error per
# case), one extended and one full pass per arm, every arm with the SAME test binary and the same seeded inputs.
# The arm "stock" (no environment) is the parent's arithmetic: without an orin-* profile the build dispatches the
# upstream SDPA kernels, as the parent does. Output: raw/<out name>/<arm>.txt (the [sdpa-error],
# [sdpa-correctness] and [sdpa-kernels] lines of both passes) and the full logs beside it.
T=$(dirname "$0"); source "$T/common.sh"; [[ $SIDE == device ]] || { echo "device only" >&2; exit 2; }
O=$A/raw/$1; BD=$A/build/$2/bundle; B=$BD/test_llama_microbench; need $B; shift 2; mkdir -p $O
{ sha256sum $B; date -u; } >> $O/env.txt
for spec in "$@"; do arm=${spec%%=*}; IFS=, read -ra E <<< "${spec#*=}"; [[ $spec == *=* ]] || E=()
  [[ -s $O/$arm.txt ]] && continue
  for tier in extended full; do
    cool_start 120
    env "${E[@]}" LD_LIBRARY_PATH=$BD $T/gl.sh $B --sdpa-correctness-only --sdpa-tier=$tier > $O/$arm-$tier.log 2>&1
    rc=$?; echo "$arm $tier rc=$rc"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && exit $rc
  done
  { echo "# arm $arm env [${E[*]}]"; grep -h '^\[sdpa-error\]\|^\[sdpa-correctness\]\|^\[sdpa-kernels\]\|^\[sarc_dev\]' $O/$arm-extended.log $O/$arm-full.log; } > $O/$arm.txt
done
echo SDPA_ERR_DONE
