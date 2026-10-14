#!/bin/bash
# probe_run.sh <session> [models=1b,3b,8b] [schemes=4w,8da4w]: next-token logits of the real-text prompt set
# (probe_prompts.py) for the four arms of stage/<session>: parent and candidate, each default and with
# ET_VK_FORCE_TILED_LINEAR=1. One logits_probe process (stage/<session>/lp) per arm and cell, each under the gpu-lab
# lock (tools/gl.sh). Output: stage/<session>/probe/{prompts-*.{txt,json},<model>-<scheme>-<arm>.{bin,log}}.
set -uo pipefail
A=${ART780M:-$HOME/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04}; S=$A/stage/$1; O=$S/probe; mkdir -p $O
T=$(cd "$(dirname "$0")" && pwd); MROOT=/mnt/linux-share/models
IFS=, read -ra MS <<< "${2:-1b,3b,8b}"; IFS=, read -ra QS <<< "${3:-4w,8da4w}"
[[ -f $O/prompts-1b.txt ]] || ~/sarc-acl/dev/executorch/.venv/bin/python $T/probe_prompts.py $O > $O/prompts.log 2>&1
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
for m in "${MS[@]}"; do IFS=: read -r MD ST <<< "${STEM[$m]}"; for q in "${QS[@]}"; do
  for arm in parent cand; do for mode in default tiled; do
    n=$m-$q-$arm-$mode; [[ -s $O/$n.bin && $(stat -c %s $O/$n.bin) == $((32 * 128256 * 4)) ]] && continue
    e=(); mapfile -t e < $S/$arm/env; [[ $mode == tiled ]] && e+=(ET_VK_FORCE_TILED_LINEAR=1)
    env "${e[@]}" LD_LIBRARY_PATH=$S $T/gl.sh $S/lp $MROOT/$MD/exported/${ST}_vulkan_$q.pte $O/prompts-$m.txt $O/$n.bin > $O/$n.log 2>&1
    echo "probe $n rc=$? prompts=$(grep -c '^prompt' $O/$n.log) $(date -u +%FT%TZ)"
  done; done
done; done
echo PROBE_DONE
