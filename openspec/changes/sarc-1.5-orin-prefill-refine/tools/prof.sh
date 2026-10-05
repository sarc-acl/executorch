#!/bin/bash
# prof.sh <out name> <build tag> [scheme:VAR:token:tile_m:tile_n:tile_k ...]: device side. In-kernel phase timing
# of linear kernels with PROF twins (shader clock; the Orin reports shaderSubgroupClock). One microbench run per
# twin (texture3d = the model path, all three models), raw outputs dumped with ET_VK_DUMP_OUTPUT_DIR into
# raw/<out name>/<scheme>/<token>/. The decode (prof_decode.py, needs numpy) writes raw/<out name>/<scheme>/phases.csv.
# Measurement only: the twins' outputs are wrong by construction. Default twin: the 4070 Ti campaign's twin of the
# zpgtr kernel both devices ship (t128x128k64g44s32mk32ra).
T=$(dirname "$0"); source "$T/common.sh"; [[ $SIDE == device ]] || { echo "device only" >&2; exit 2; }
O=$A/raw/$1; BD=$A/build/$2/bundle; B=$BD/test_llama_microbench; need $B; shift 2
[[ $# -gt 0 ]] || set -- 8da4w:ET_VK_SARC_DQ8CA_VARIANT:t128x128k64g44s32mk32rap:128:128:64
for spec in "$@"; do IFS=: read -r q var tok tm tn tk <<< "$spec"
  d=$O/$q/$tok; mkdir -p $d
  LD_LIBRARY_PATH=$BD $B --list --linear --regime=prefill --scheme=$q --storage=texture3d > $d/cases.txt 2>&1
  cool_start 120
  env $var=$tok ET_VK_DUMP_OUTPUT_DIR=$d LD_LIBRARY_PATH=$BD $T/gl.sh $B --linear --regime=prefill --scheme=$q --storage=texture3d --skip-correctness > $d/run.log 2>&1
  rc=$?; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && exit $rc
  { echo "# $tok rc=$rc dumps=$(ls $d/out_*.bin 2>/dev/null | wc -l)"; python3 $T/prof_decode.py $d $tm $tn $tk; } >> $O/$q/phases.csv
done
echo PROF_DONE
