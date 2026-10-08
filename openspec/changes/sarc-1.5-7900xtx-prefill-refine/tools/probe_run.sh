#!/bin/bash
# probe_run.sh <session> [models=1b,3b,8b] [schemes=4w,8da4w]: next-token logits of the real-text prompt set
# (probe_prompts.py, results/7900xtx/probe/prompts-*.txt) for the four arms of stage/<session>: parent and candidate,
# each default and with ET_VK_FORCE_TILED_LINEAR=1. One logits_probe process (stage/<session>/lp, from the
# candidate's build: the probe is a runner, the kernels come from the backend it links, the same for both arms)
# per arm and cell, each a gl.sh job. Output: stage/<session>/probe/<model>-<scheme>-<arm>-<mode>.{bin,log}.
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
S=$A/stage/$1; O=$S/probe; mkdir -p $O; P=$T/../results/7900xtx/probe
IFS=, read -ra MS <<< "${2:-1b,3b,8b}"; IFS=, read -ra QS <<< "${3:-4w,8da4w}"
declare -A STEM=([1b]=llama3_2-1b [3b]=llama3_2-3b [8b]=llama3_1-8b)
for m in "${MS[@]}"; do for q in "${QS[@]}"; do cat $MFLAT/${STEM[$m]}_vulkan_$q.pte > /dev/null
  for arm in parent cand; do for mode in default tiled; do
    n=$m-$q-$arm-$mode; [[ -s $O/$n.bin && $(stat -c %s $O/$n.bin) == $((32 * 128256 * 4)) ]] && continue
    e=(); mapfile -t e < $S/$arm/env; [[ $mode == tiled ]] && e+=(ET_VK_FORCE_TILED_LINEAR=1)
    env "${e[@]}" $T/gl.sh $S/lp $MFLAT/${STEM[$m]}_vulkan_$q.pte $P/prompts-$m.txt $O/$n.bin > $O/$n.log 2>&1
    echo "probe $n rc=$? prompts=$(grep -c '^prompt' $O/$n.log) $(date -u +%FT%TZ)"
  done; done
done; done
echo PROBE_DONE
