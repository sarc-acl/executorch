#!/bin/bash
# probe_run.sh <build tag> <arm label> [VAR=VALUE ...]: device side. Full last-position logits of the 41-prompt
# set (the sibling campaign's set, results/orin/probe/broad/prompts_ids.txt, unchanged: 32 tile-aligned lengths
# 64 .. 2048, 8 unaligned lengths and the gate's unaligned prompt) for all six cells with the given build and
# environment, one logits_dump process per cell, under the gpu-lab lock.
# Output: probe/broad/<arm label>/<model>-<scheme>.{bin,log}. logits_dump is cross-built per build tag
# (build-extra.sh). Resumable: a cell whose log ends with "prompts 41" is skipped.
T=$(dirname "$0"); source "$T/common.sh"; [[ $SIDE == device ]] || { echo "device only" >&2; exit 2; }
BT=$1; ARM=$2; shift 2; P=$A/probe; IDS=$CHANGE/results/orin/probe/broad/prompts_ids.txt; need $IDS
B=$A/build/$BT/bundle/logits_dump; need $B
O=$P/broad/$ARM; mkdir -p $O; cp -f $IDS $CHANGE/results/orin/probe/broad/prompts_meta.csv $P/broad/
{ echo "build $BT env [$*]"; sha256sum $B $IDS; date -u; } >> $O/env.txt
declare -A STEM=([1b]=llama3_2-1b [3b]=llama3_2-3b [8b]=llama3_1-8b)
for m in 1b 3b 8b; do ST=${STEM[$m]}; for q in 4w 8da4w; do
  tail -1 $O/$m-$q.log 2>/dev/null | grep -q '^prompts 41$' && continue
  cool_start 120; echo "pre $(mem_line)" > $O/$m-$q.mem
  env "$@" $T/gl.sh $B $MODELDIR/${ST}_vulkan_$q.pte $IDS $O/$m-$q.bin > $O/$m-$q.log 2>&1
  rc=$?; echo "post $(mem_line)" >> $O/$m-$q.mem; echo "$ARM $m $q rc=$rc $(tail -1 $O/$m-$q.log)"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && exit $rc
done; done
echo PROBE_RUN_DONE
