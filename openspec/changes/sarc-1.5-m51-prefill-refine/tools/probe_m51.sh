#!/bin/bash
# probe_m51.sh <session> [models=1b,3b] [schemes=4w,8da4w]: next-token logits of the real-text prompt set
# (probe_prompts.py) for the four arms of stage/m51-LOCAL-ONLY/<session>: parent and candidate (their build's
# logits_probe and env), each default and with ET_VK_FORCE_TILED_LINEAR=1. Adapted from the 780M's probe_run.sh.
# Output: <stage>/probe/{prompts-*.{txt,json},<model>-<scheme>-<arm>-<mode>.{bin,log}}; resumable.
# One coordinator-hold unit; device-state guard before each run.
set -uo pipefail
[[ -n ${SARC_HOLD_UNIT:-} ]] || exec env SARC_HOLD_UNIT=1 "$(dirname "$(readlink -f "$0")")/hold.sh" run "probe set probe_m51.sh $*" "$0" "$@"
source "$(dirname "$(readlink -f "$0")")/dev.sh"
SES=$1; S=$ART/stage/$LOC/$SES; DS=$DEV_ROOT/stage/$SES; O=$S/probe; mkdir -p "$O"
IFS=, read -ra MS <<< "${2:-1b,3b}"; IFS=, read -ra QS <<< "${3:-4w,8da4w}"
declare -A STEM=([1b]=llama3_2_1b [3b]=llama3_2_3b [8b]=llama3_1_8b)
[[ -f $O/prompts-1b.txt ]] || "$ART/venv/m51/bin/python" "$TOOLS/probe_prompts.py" "$O" > "$O/prompts.log" 2>&1
A shell "mkdir -p $DS/probe" < /dev/null
for arm in parent cand; do
  tag=$(sed -n 's/^\(parent\|cand  \) = build \(\S*\) .*/\1 \2/p' "$S/STAGE.md" | awk -v a=$arm '$1 == a {print $2}')
  A push "$ART/build/m51/$tag/probe/logits_probe" "$DS/probe/lp-$arm" > /dev/null; A shell "chmod 755 $DS/probe/lp-$arm" < /dev/null
  echo "$arm logits_probe from build $tag $(sha256sum "$ART/build/m51/$tag/probe/logits_probe" | cut -c1-16)" >> "$O/env.txt"
done
for m in "${MS[@]}"; do A push "$O/prompts-$m.txt" "$DS/probe/" > /dev/null; done
for m in "${MS[@]}"; do for q in "${QS[@]}"; do for arm in parent cand; do for mode in default tiled; do
  n=$m-$q-$arm-$mode; [[ -s $O/$n.bin && $(stat -c %s "$O/$n.bin") == $((32 * 128256 * 4)) ]] && continue
  st=$(device_state); [[ $st == ok ]] || { echo "board not fit before $n: $st"; exit 3; }
  e=$(tr '\n' ' ' < "$S/$arm/env"); [[ $mode == tiled ]] && e+=" ET_VK_FORCE_TILED_LINEAR=1"
  A shell "cd $DS/probe && rm -f $n.bin && $e timeout 1190 ./lp-$arm $DEV_ROOT/models/${STEM[$m]}_${q}_embq_ctx3072.pte prompts-$m.txt $n.bin < /dev/null > $n.log 2>&1; echo RC=\$? >> $n.log" < /dev/null > /dev/null 2>&1
  alive || { echo "board gone during probe $n $(date -u +%FT%TZ)" | tee -a "$ART/ABORTED"; exit 3; }
  A pull "$DS/probe/$n.log" "$O/" > /dev/null; A pull "$DS/probe/$n.bin" "$O/" > /dev/null 2>&1
  echo "probe $n $(tail -1 "$O/$n.log") prompts=$(grep -c '^prompt' "$O/$n.log") $(date -u +%FT%TZ)"
done; done; done; done
"$ART/venv/m51/bin/python" "$TOOLS/probe_compare.py" "$O" "$O/compare.csv" > "$O/compare.out" 2>&1; tail -1 "$O/compare.out"
echo PROBE_DONE
