#!/bin/bash
# run_nosel.sh: the "with nothing selected the build behaves as its parent" check, runtime half. The parent build
# (tuning campaign's topic7, same backend sources as this branch's parent a434a4962) and this experiment's build
# (exp1) each run the prefill linear suite (4w and 8da4w, texture3d, all models) twice: with no SARC variable at
# all, and with the campaign's final configuration (ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-fused1).
# data/nosel/<build>-<cfg>.json; the dispatched kernel per case is compared by tools/nosel_compare.py.
# Timings of these processes are not used.
. "$(dirname "$(readlink -f "$0")")/guard.sh"; BOX=$(cd $ROOT/../../../.. && pwd); O=$ROOT/data/nosel; mkdir -p $O
export TMPDIR=$BOX/.tmp ETVK_DEVICE_INDEX=0; unset ET_VK_SARC_DEV_PROFILE ET_VK_SARC_DQ8CA_VARIANT ET_VK_SARC_EXP ET_VK_SARC_UNVERIFIED ET_VK_SARC_Q4GSW_VARIANT
declare -A B=([parent]=/mnt/linux-share/hmz-campaigns/b580-fused/.artifacts/build/topic7/tests/test_llama_microbench [exp1]=$BOX/.artifacts/build/exp1/tests/test_llama_microbench)
for b in parent exp1; do for c in none fused1; do
  E=(); [[ $c == fused1 ]] && E=(ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-fused1)
  gpu_job nosel-$b-$c env "${E[@]}" timeout 900 ${B[$b]} --linear --regime=prefill --storage=texture3d --skip-correctness --json-out=$O/$b-$c.json > $O/$b-$c.log 2>&1
  echo "$? $(sha256sum ${B[$b]} | cut -c1-16)" > $O/$b-$c.rc
done; done
python3 $ROOT/tools/nosel_compare.py $O > $O/compare.txt 2>&1; tail -3 $O/compare.txt
