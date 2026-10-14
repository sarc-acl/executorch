#!/bin/bash
# phases.sh <session> <tag> <family 4w|8da4w> <prof kernel base> <tile_m> <tile_n>: in-kernel phase timing (shader clock) of a PROF twin on
# the twelve real texture3d prefill shapes, with the test_llama_microbench of stage/<session>: the twin is selected by exact name
# (ET_VK_SARC_780M_{Q4,DQ}) and ET_VK_DUMP_OUTPUT_DIR; the twin writes its counters over its output (wrong results by design).
#   on the GPU host:           runs the job (one gl.sh job) into stage/<session>/phases/<tag>/
#   on the control workstation: decodes the pulled dumps (prof_decode.py, numpy) into results/7900xtx/phases/<tag>.csv
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$1; TAG=$2; F=$3; K=$4; TM=$5; TN=$6; O=$A/stage/$S/phases/$TAG; mkdir -p $O
declare -A V=([4w]=ET_VK_SARC_780M_Q4 [8da4w]=ET_VK_SARC_780M_DQ)
if [[ $WHERE == gpu ]]; then
  B=$A/stage/$S/test_llama_microbench
  $B --list --linear --regime=prefill --scheme=$F --storage=texture3d > $O/cases.txt 2>&1
  env ET_VK_SARC_UNVERIFIED=1 ${V[$F]}=$K ET_VK_DUMP_OUTPUT_DIR=$O $T/gl.sh $B --linear --regime=prefill --scheme=$F \
    --storage=texture3d --skip-correctness --json-out=$O/run.json > $O/run.log 2>&1
  echo "phases $TAG rc=$?"
else
  mkdir -p $T/../results/7900xtx/phases
  $PY $T/prof_decode.py $O $TM $TN > $T/../results/7900xtx/phases/$TAG.csv 2> $O/decode.err; cat $T/../results/7900xtx/phases/$TAG.csv
fi
