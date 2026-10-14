#!/bin/bash
# probe_run.sh <build tag> <arm label> [VAR=VALUE ...]: full last-position logits of the 41-prompt set
# (probe_prompts.py, probe/broad/prompts_ids.txt) for all six cells with the given build and environment, one
# logits_dump process per cell, under the gpu-lab lock. Output: probe/broad/<arm label>/<model>-<scheme>.{bin,log}.
# The tool is built once per build tag in the build container against that tag's install prefix
# (probe/dump-<tag>/logits_dump). Resumable: a cell whose log ends with "prompts 41" is skipped.
T=$(dirname "$0"); source "$T/common.sh"; BT=$1; ARM=$2; shift 2; P=$A/probe; IDS=$P/broad/prompts_ids.txt; need $IDS
B=$P/dump-$BT/logits_dump
if [[ ! -x $B ]]; then
  SRC=$P/dump-src; mkdir -p $SRC; cp -f $T/logits_dump/main.cpp $T/logits_dump/CMakeLists.txt $SRC/
  docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $R:$R localhost/et-vk-build:rocky10 bash -c "cmake $SRC -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_ROOT=$A/src/$BT/executorch -DCMAKE_PREFIX_PATH=$A/build/$BT/llama -DCMAKE_FIND_ROOT_PATH=$A/build/$BT/llama -B$P/dump-$BT && cmake --build $P/dump-$BT -j12" > $P/dump-$BT.log 2>&1 || { echo "logits_dump build failed, see $P/dump-$BT.log" >&2; exit 3; }
fi
O=$P/broad/$ARM; mkdir -p $O; { echo "build $BT env [$*]"; sha256sum $B $IDS; date -u; } >> $O/env.txt
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
for m in 1b 3b 8b; do IFS=: read -r MD ST <<< "${STEM[$m]}"; for q in 4w 8da4w; do
  tail -1 $O/$m-$q.log 2>/dev/null | grep -q '^prompts 41$' && continue
  cool_start 60 120
  env "$@" $T/gl.sh $B /mnt/linux-share/models/$MD/exported/${ST}_vulkan_$q.pte $IDS $O/$m-$q.bin > $O/$m-$q.log 2>&1
  rc=$?; echo "$ARM $m $q rc=$rc $(tail -1 $O/$m-$q.log)"; [[ $rc == 70 || $rc == 75 || $rc == 76 ]] && exit $rc
done; done
echo PROBE_RUN_DONE
