#!/bin/bash
# phases3.sh <microbench binary> <env file> <tag> <scheme 4w|8da4w> <prof kernel base> <tile_m> <tile_n> <tile_k>: like phases2.sh for either scheme
# (exact-name selection through ET_VK_SARC_780M_Q4 / ET_VK_SARC_780M_DQ). Output $A/raw/phases/<tag>/ and results/rx7600/round2/phases/<tag>.csv.
source "$(dirname "$(readlink -f "$0")")/env.sh"
B=$1; BASE=$2; TAG=$3; Q=$4; K=$5; TM=$6; TN=$7; TK=$8; O=$A/raw/phases/$TAG; mkdir -p $O $T/../results/rx7600/round2/phases
declare -A V=([4w]=ET_VK_SARC_780M_Q4 [8da4w]=ET_VK_SARC_780M_DQ)
$B --list --linear --regime=prefill --scheme=$Q --storage=texture3d > $O/cases.txt 2>&1
env $(cat "$BASE") ${V[$Q]}=$K ET_VK_DUMP_OUTPUT_DIR=$O $T/gl.sh $B --linear --regime=prefill --scheme=$Q \
  --storage=texture3d --skip-correctness --json-out=$O/run.json > $O/run.log 2>&1
echo "run rc=$?"
<toolchain-share>/Python-3.12.9-1/bin/python3 -I $T/prof_decode2.py $O $TM $TN $TK > $T/../results/rx7600/round2/phases/$TAG.csv 2> $O/decode.err
cat $T/../results/rx7600/round2/phases/$TAG.csv
