#!/bin/bash
# phases.sh <tag> <family 4w|8da4w> <prof kernel base> <tile_m> <tile_n>: in-kernel phase timing (shader clock) of a
# PROF twin on the twelve real texture3d prefill shapes: test_llama_microbench --linear with the twin selected by
# exact name (ET_VK_SARC_780M_{Q4,DQ}) and ET_VK_DUMP_OUTPUT_DIR, decoded by prof_decode.py (the twin writes its
# counters over its output: wrong results by design). Output $A/raw/phases/<tag>/ and results/rx7600/phases/<tag>.csv.
source "$(dirname "$(readlink -f "$0")")/env.sh"
TAG=$1; F=$2; K=$3; TM=$4; TN=$5; O=$A/raw/phases/$TAG; mkdir -p $O
declare -A V=([4w]=ET_VK_SARC_780M_Q4 [8da4w]=ET_VK_SARC_780M_DQ)
B=$A/build/rx7600/parent/tests/test_llama_microbench
$B --list --linear --regime=prefill --scheme=$F --storage=texture3d > $O/cases.txt 2>&1
env ET_VK_SARC_UNVERIFIED=1 ${V[$F]}=$K ET_VK_DUMP_OUTPUT_DIR=$O $T/gl.sh $B --linear --regime=prefill --scheme=$F \
  --storage=texture3d --skip-correctness --json-out=$O/run.json > $O/run.log 2>&1
echo "run rc=$?"
<toolchain-share>/Python-3.12.9-1/bin/python3 $T/prof_decode.py $O $TM $TN > $T/../results/rx7600/phases/$TAG.csv 2> $O/decode.err
cat $T/../results/rx7600/phases/$TAG.csv
