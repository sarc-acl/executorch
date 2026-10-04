#!/bin/bash
# prof.sh <out name> <build tag> <scheme: 4w|8da4w> <prof tile token> [storage=texture3d] [model=llama-3.2-1b]:
# in-kernel phase timing of one linear kernel with its MEASUREMENT-ONLY sarc_dev_prof_* twin (shader clock).
# Runs test_llama_microbench --linear --regime=prefill with the prof variant and ET_VK_DUMP_OUTPUT_DIR, then
# prof_decode.py. Output in raw/<out name>/: out_*.bin, cases.txt, run.log, phases.csv. The prof kernels write
# phase counters over their output, so nothing here is a correctness or speed result.
. "$(dirname "$(readlink -f "$0")")/host.sh"; O=$A/raw/$1; B=$A/build/$2/tests/test_llama_microbench; Q=$3; TOK=$4; ST=${5:-texture3d}; MODEL=${6:-llama-3.2-1b}
mkdir -p $O || exit 2; VAR=ET_VK_SARC_Q4GSW_VARIANT; [[ $Q == 8da4w ]] && VAR=ET_VK_SARC_DQ8CA_VARIANT
[[ $TOK =~ t([0-9]+)x([0-9]+)k([0-9]+) ]] || { echo "tile token?" >&2; exit 2; }; TM=${BASH_REMATCH[1]}; TN=${BASH_REMATCH[2]}; TK=${BASH_REMATCH[3]}
$B --list --linear --regime=prefill --scheme=$Q --storage=$ST --model=$MODEL > $O/cases.txt 2>&1
cool_start
env $VAR=$TOK ET_VK_DUMP_OUTPUT_DIR=$O $TOOLS/gl.sh $B --linear --regime=prefill --scheme=$Q --storage=$ST --model=$MODEL --skip-correctness > $O/run.log 2>&1; rc=$?
echo "prof $Q $TOK rc=$rc $(grep -o 'sarc_dev_prof[a-z0-9_]*' $O/run.log | sort -u | tr '\n' ' ')"
[[ $rc == 75 || $rc == 76 ]] && exit $rc
$XE2_PYTHON $TOOLS/prof_decode.py $O $TM $TN 2048 $TK | tee $O/phases.csv
