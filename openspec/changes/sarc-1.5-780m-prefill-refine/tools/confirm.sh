#!/bin/bash
# confirm.sh <artifact dir> <family: 4w|8da4w> <manifest,manifest,...> <screen.csv>... : the confirmation of a
# family after its screening (owner decision 2026-10-04: confirm the best 10 per shape with the full measurement).
#   1. every configuration within 8 % of the fastest on some screened shape, once, full measurement (all twelve
#      shapes, 3 warm-up + 5 timed runs): raw/confirm-<family>/full-r1.csv
#   2. the 10 fastest per shape of that, four more times (5 repeats in all): full-r2..r5.csv
#   3. those configurations, 12 passes of --production-diff on the three models (texture3d, M = 2048, sampled
#      reference; 8da4w with non-zero zero-points as in verify.sh): pdiff/<token>-<model>-r<i>.log, pdiff.csv
# One GPU job at a time under the gpu-lab lock; resumable (every step skips what is already there).
set -uo pipefail
D=$(realpath "$1"); FAM=$2; MAN=$3; shift 3; T=$(cd "$(dirname "$0")" && pwd)
O=$D/raw/confirm-$FAM; mkdir -p $O/pdiff $D/space/confirm-$FAM; cd $D
declare -A ALWAYS=([4w]=t128x128k32g42s32f32c,t128x256k32g42s32f32c [8da4w]=t128x64k32g42s32,t128x64k32g22s32,bt_t128x64k32g22s32)
declare -A ENVV=([4w]=ET_VK_SARC_780M_Q4 [8da4w]=ET_VK_SARC_780M_DQ)
[[ -f space/confirm-$FAM/all.csv ]] || python3 $T/confirm_select.py within 8 space/confirm-$FAM/all.csv $MAN "$@" --always ${ALWAYS[$FAM]} 2> $O/select-all.txt
python3 $T/sweep_space.py $D space/confirm-$FAM/all.csv $O/full-r1.csv --only $FAM --ignore-pause --tmax 62 > $O/full-r1.out 2>&1
[[ -f space/confirm-$FAM/top.csv ]] || python3 $T/confirm_select.py top 10 space/confirm-$FAM/top.csv $MAN $O/full-r1.csv --always ${ALWAYS[$FAM]} 2> $O/select-top.txt
for i in 2 3 4 5; do
  python3 $T/sweep_space.py $D space/confirm-$FAM/top.csv $O/full-r$i.csv --only $FAM --ignore-pause --tmax 62 > $O/full-r$i.out 2>&1
done
python3 $T/confirm_summary.py $O/summary.csv $O/full-r[1-5].csv > $O/summary.txt 2>&1
ZP=(); [[ $FAM == 8da4w ]] && ZP=(--production-diff-nonzero-zp)
[[ -f $O/pdiff.csv ]] || echo "token,model,pass,rc,shapes_passed,shapes_on_kernel,verdict" > $O/pdiff.csv
tail -n +2 space/confirm-$FAM/top.csv | while IFS=, read -r batch fam stage token kb heads; do
  cp -f bin/microbench-$batch $O/pdiff/mb-$batch
  for m in llama-3.2-1b llama-3.2-3b llama-3.1-8b; do for i in $(seq 1 12); do
    grep -q "^$token,$m,$i," $O/pdiff.csv && continue
    L=$O/pdiff/$token-$m-r$i.log
    env ${ENVV[$FAM]}=$kb $T/gl.sh $O/pdiff/mb-$batch --production-diff --production-diff-model=$m --production-diff-op=$FAM \
      --production-diff-storage=texture3d "${ZP[@]}" > $L 2>&1 < /dev/null; rc=$?
    echo "$token,$m,$i,$rc,$(grep -c 'correctness=PASSED' $L),$(grep 'correctness=' $L | grep -c -- "-> ${kb}_"),$(grep -o 'ALL PASSED\|FAILED' $L | tail -1)" >> $O/pdiff.csv
  done; done
done
echo "CONFIRM_DONE $(date -u +%FT%TZ)" > $O/done
