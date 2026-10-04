#!/bin/bash
# prof.sh <out name> <build tag>: in-kernel phase timing of the shipped linear kernels with the PROF twins
# (gen_4070ti_prof.py). One microbench run per twin (texture3d = the model path, all three models), raw outputs
# dumped with ET_VK_DUMP_OUTPUT_DIR and decoded with prof_decode.py into raw/<out name>/<scheme>/phases.csv
# (one block per twin; decode needs numpy: TRACE_PY). Measurement only: the twins' outputs are wrong by construction.
T=$(dirname "$0"); source "$T/common.sh"; O=$A/raw/$1; B=$A/build/$2/tests/test_llama_microbench; need $B
run() { # run <scheme> <env var> <token> <tile m> <tile n> <tile k>
  local d=$O/$1/$3; mkdir -p $d
  $B --list --linear --regime=prefill --scheme=$1 --storage=texture3d > $d/cases.txt 2>&1
  env $2=$3 ET_VK_DUMP_OUTPUT_DIR=$d $T/gl.sh $B --linear --regime=prefill --scheme=$1 --storage=texture3d --skip-correctness > $d/run.log 2>&1
  local rc=$?; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && exit $rc
  { echo "# $3 rc=$rc dumps=$(ls $d/out_*.bin 2>/dev/null | wc -l)"; ${TRACE_PY:-python3} $T/prof_decode.py $d $4 $5 $6; } >> $O/$1/phases.csv
}
mkdir -p $O/4w $O/8da4w; : > $O/4w/phases.csv; : > $O/8da4w/phases.csv
run 4w ET_VK_SARC_Q4GSW_VARIANT t256x128k16g42s32gap 256 128 16
run 4w ET_VK_SARC_Q4GSW_VARIANT t128x128k16g24s32gap 128 128 16
run 8da4w ET_VK_SARC_DQ8CA_VARIANT t128x128k64g44s32mk32rap 128 128 64
echo PROF_DONE
