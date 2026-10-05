#!/bin/bash
# deploy.sh [build tag ...]: workstation side. Copies to the device (~/hmz-sarc-orin/, nothing else there is touched):
#   executorch/<this change>/tools/      this directory (scripts only)
#   executorch/<this change>/results/orin/clkmin.json   when it exists
#   executorch/sarc/tools/verify.sh      UNMODIFIED: must be byte-identical to the parent commit's file
#   executorch/<kit>/prompts/            the kit prompts (hashes checked against the commit)
#   build/<tag>/bundle/ and build/<tag>.src.txt   for each build tag given (--extra <tag>...: only logits_dump)
# and makes sure the gpu-lab lock file exists and is writable.
set -euo pipefail
source "$(dirname "$0")/common.sh"; [[ $SIDE == ws ]] || { echo "workstation only" >&2; exit 2; }
V=sarc/tools/verify.sh
[[ $(git -C $ET rev-parse $PARENT_COMMIT:$V) == $(git -C $ET hash-object $ET/$V) ]] || { echo "verify.sh differs from the parent commit" >&2; exit 77; }
for f in prompt_2048.txt prompt_check.txt prompt_real_2048.txt; do K=openspec/changes/sarc-1.5-e2e-benchmark/kit/prompts/$f
  [[ $(git -C $ET rev-parse $PARENT_COMMIT:$K) == $(git -C $ET hash-object $ET/$K) ]] || { echo "$f differs from the parent commit" >&2; exit 77; }; done
D=$DEVROOT/executorch
dssh "mkdir -p $D/$CHANGE_REL/results/orin/probe/broad $D/sarc/tools $D/${KIT#$ET/}/prompts $DEVROOT/build $DEVROOT/jobs ~/.cache/gpu-lab && touch ~/.cache/gpu-lab/lock-$LOCK && test -w ~/.cache/gpu-lab/lock-$LOCK"
rsync -a --delete --exclude jetson-cross --exclude __pycache__ $TOOLS/ $DEVICE:$D/$CHANGE_REL/tools/
[[ -f $CHANGE/results/orin/clkmin.json ]] && rsync -a $CHANGE/results/orin/clkmin.json $DEVICE:$D/$CHANGE_REL/results/orin/
[[ -d $CHANGE/results/orin/probe/broad ]] && rsync -a --include "prompts_*" --exclude "*" $CHANGE/results/orin/probe/broad/ $DEVICE:$D/$CHANGE_REL/results/orin/probe/broad/
rsync -a $ET/$V $DEVICE:$D/sarc/tools/; rsync -a $KIT/prompts/ $DEVICE:$D/${KIT#$ET/}/prompts/
EXTRA=0; [[ ${1:-} == --extra ]] && { EXTRA=1; shift; }
for t in "$@"; do need $A/build/$t.src.txt $A/build/$t/bundle/llama_main
  if [[ $EXTRA == 1 ]]; then   # only the logits_dump added by build-extra.sh
    need $A/build/$t/bundle/logits_dump; rsync -a $A/build/$t/bundle/logits_dump $DEVICE:$DEVROOT/build/$t/bundle/
    h=$(dssh "sha256sum $DEVROOT/build/$t/bundle/logits_dump" | cut -d' ' -f1); grep -q "^$h .*/bundle/logits_dump\$" $A/build/$t.src.txt || { echo "hash mismatch: $t logits_dump" >&2; exit 4; }
    echo "deployed logits_dump of $t"; continue; fi
  dssh "test ! -e $DEVROOT/build/$t.src.txt" || { echo "build $t already on the device (tags are immutable)"; continue; }
  rsync -a $A/build/$t/bundle $DEVICE:$DEVROOT/build/$t/; rsync -a $A/build/$t.src.txt $DEVICE:$DEVROOT/build/
  dssh "cd $DEVROOT/build/$t/bundle && sha256sum llama_main libllama_runner.so test_llama_microbench" | while read -r h f; do
    grep -q "^$h .*/bundle/$f\$" $A/build/$t.src.txt || { echo "hash mismatch on the device: $t $f" >&2; exit 4; }; done
  echo "deployed build $t"
done
dssh "sha256sum $D/sarc/tools/verify.sh | cut -c1-16; ls $DEVROOT/build"
