#!/bin/bash
# phase_m51.sh <mbstage name> <4w token of a PROF twin> <tile_m> <tile_n> <tile_k> [rounds=3]: in-kernel phase timing.
# Runs test_llama_microbench --linear --regime=prefill --scheme=4w --storage=texture3d --skip-correctness with the
# PROF twin selected (its output carries shader-clock phase counters, wrong by construction) and
# ET_VK_DUMP_OUTPUT_DIR on the board; pulls the dumps and decodes them (prof_decode.py) into
# <stage>/phase/<token>-r<i>.csv. One coordinator-hold unit per round; the board cools before each round.
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/dev.sh"
STG=$ART/stage/$LOC/$1; TOK=$2; TM=$3; TN=$4; TK=$5; R=${6:-3}; DD=$DEV_ROOT/stage/$1/top/dump
cd "$STG" || exit 2; mkdir -p phase
for ((r = 1; r <= R; r++)); do
  t0=$SECONDS; while (( $(gtemp) > ${START_C:-38} && SECONDS - t0 < 300 )); do sleep 10; done
  A shell "rm -rf $DD && mkdir -p $DD" < /dev/null
  "$TOOLS/hold.sh" run "phase $TOK r$r" env ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_Q4GSW_VARIANT=$TOK ET_VK_DUMP_OUTPUT_DIR=$DD \
    timeout 1800 ./test_llama_microbench --linear --regime=prefill --scheme=4w --storage=texture3d --skip-correctness \
    > phase/$TOK-r$r.log 2>&1
  rm -rf phase/dump-r$r; A pull "$DD" "phase/dump-r$r" > /dev/null 2>&1
  grep -o 'llama-[^ ]*_linear_q4gsw_M[0-9]*_K[0-9]*_N[0-9]*[^ ]*' phase/$TOK-r$r.log | sort -u > phase/dump-r$r/cases.txt
  "$ART/venv/m51/bin/python" "$TOOLS/prof_decode.py" phase/dump-r$r $TM $TN 2048 $TK > phase/$TOK-r$r.csv 2>&1
  echo "phase $TOK r$r: $(grep -c . phase/$TOK-r$r.csv) lines, kernels: $(grep -o 'sarc_dev_prof_q4gsw[a-z0-9_]*' phase/$TOK-r$r.log | sort | uniq -c | tr '\n' ' ')"
done
echo PHASE_DONE
