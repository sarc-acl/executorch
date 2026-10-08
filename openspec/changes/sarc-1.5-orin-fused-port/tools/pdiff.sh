#!/bin/bash
# pdiff.sh <out name> <build tag> <scheme> <models: 1b,3b,8b> <token...>: device side. The sampled production-diff
# of test_llama_microbench (the same invocation verify.sh makes: texture3d, non-zero zero points for 8da4w,
# unchanged tolerances) for linear kernels selected by name (ET_VK_SARC_Q4GSW_VARIANT / ET_VK_SARC_DQ8CA_VARIANT),
# before a kernel is put into a profile. Token "base" = no override. One log per (token, model) in
# raw/<out name>/ and one summary line each: rc, the final verdict line, the kernels dispatched.
T=$(dirname "$0"); source "$T/common.sh"; [[ $SIDE == device ]] || { echo "device only" >&2; exit 2; }
O=$A/raw/$1; BD=$A/build/$2/bundle; B=$BD/test_llama_microbench; Q=$3; IFS=, read -ra MS <<< "$4"; shift 4; need $B; mkdir -p $O
declare -A MD=([1b]=llama-3.2-1b [3b]=llama-3.2-3b [8b]=llama-3.1-8b)
VAR=ET_VK_SARC_Q4GSW_VARIANT; Z=""; [[ $Q == 8da4w ]] && { VAR=ET_VK_SARC_DQ8CA_VARIANT; Z=--production-diff-nonzero-zp; }
for t in "$@"; do for m in "${MS[@]}"; do L=$O/$Q-$t-$m.log
  [[ -s $L ]] && grep -q '^\[production-diff\] \(ALL PASSED\|FAILED\)' $L && continue
  E=(); [[ $t != base ]] && E=("$VAR=$t")
  cool_start 60
  env "${E[@]}" LD_LIBRARY_PATH=$BD $T/gl.sh $B --production-diff --production-diff-model=${MD[$m]} --production-diff-op=$Q --production-diff-storage=texture3d $Z > $L 2>&1
  rc=$?; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && exit $rc
  echo "pdiff $Q $t $m rc=$rc $(grep -E '^\[production-diff\] (ALL PASSED|FAILED)' $L | tail -1 | cut -c1-90) kernels: $(grep -o 'sarc_[a-z0-9_]*\|linear_[a-z0-9_]*tiled[a-z0-9_]*\|q4gsw_linear[a-z0-9_]*' $L | sort | uniq -c | sort -rn | head -2 | tr -s ' ' | tr '\n' ';')" | tee -a $O/summary.txt
done; done
echo PDIFF_DONE
