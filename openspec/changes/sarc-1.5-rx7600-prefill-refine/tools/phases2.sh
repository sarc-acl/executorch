#!/bin/bash
# phases2.sh <microbench binary> <env file> <tag> <prof kernel base> <tile_m> <tile_n> <tile_k>: in-kernel phase timing (shader clock)
# of a PROF variant on the twelve real 8da4w texture3d prefill shapes (round 2). Like phases.sh, with the binary and the base environment
# as arguments and the K step passed to the decoder. Output $A/raw/phases/<tag>/ and results/rx7600/round2/phases/<tag>.csv.
source "$(dirname "$(readlink -f "$0")")/env.sh"
B=$1; BASE=$2; TAG=$3; K=$4; TM=$5; TN=$6; TK=$7; O=$A/raw/phases/$TAG; mkdir -p $O $T/../results/rx7600/round2/phases
$B --list --linear --regime=prefill --scheme=8da4w --storage=texture3d > $O/cases.txt 2>&1
env $(cat "$BASE") ET_VK_SARC_780M_DQ=$K ET_VK_DUMP_OUTPUT_DIR=$O $T/gl.sh $B --linear --regime=prefill --scheme=8da4w \
  --storage=texture3d --skip-correctness --json-out=$O/run.json > $O/run.log 2>&1
echo "run rc=$?"
<toolchain-share>/Python-3.12.9-1/bin/python3 -I $T/prof_decode2.py $O $TM $TN $TK > $T/../results/rx7600/round2/phases/$TAG.csv 2> $O/decode.err
cat $T/../results/rx7600/round2/phases/$TAG.csv
