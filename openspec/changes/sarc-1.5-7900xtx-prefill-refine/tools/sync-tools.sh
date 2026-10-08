#!/bin/bash
# sync-tools.sh: put this tools directory, the unmodified sarc/tools/verify.sh and the flat model layout on the GPU host.
#   <gpu-root>/{tools,sarc-tools/verify.sh,models/,stage/,logs/,tmp/,env.gpu}; the marker tools/ON_GPU_HOST makes env.sh behave as the GPU host.
# Idempotent; never touches anything outside <gpu-root>. Run from the control workstation.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"; source $T/rlib.sh
[[ $WHERE == ws ]] || { echo "run on the control workstation" >&2; exit 2; }
M=${GMODELS:-$(dirname "$GROOT")/et-models}
clog "rsync tools + verify.sh to <gpu-root>; flat model links"
rsh "mkdir -p $GROOT/{tools,sarc-tools,models,stage,logs,tmp}"
rsync -a --delete --exclude=__pycache__ -e "ssh -o BatchMode=yes" "$T/" "$GPUHOST:$GROOT/tools/"
rsh "mkdir -p $GROOT/results/7900xtx/probe"
rsync -a -e "ssh -o BatchMode=yes" "$T/../results/7900xtx/probe/" "$GPUHOST:$GROOT/results/7900xtx/probe/"   # probe prompt sets (same as the 780M's)
rsync -a -e "ssh -o BatchMode=yes" "$ET/sarc/tools/verify.sh" "$GPUHOST:$GROOT/sarc-tools/verify.sh"
rsh "touch $GROOT/tools/ON_GPU_HOST; printf 'LOCK=%s\nGCACHE=%s\n' '${GLOCK:-7900xtx-gpu-host}' '$(dirname "$GROOT")/.cache' > $GROOT/env.gpu
mkdir -p $(dirname "$GROOT")/.cache/gpu-lab; touch $(dirname "$GROOT")/.cache/gpu-lab/lock-${GLOCK:-7900xtx-gpu-host}
cd $GROOT/models
for x in 1b:llama3_2-1b:llama3_2_1b 3b:llama3_2-3b:llama3_2_3b 8b:llama3_1-8b:llama3_1_8b; do IFS=: read -r k stem real <<< \"\$x\"
  for q in 4w 8da4w; do ln -sfn $M/\${real}_\${q}_embq_ctx3072.pte \${stem}_vulkan_\${q}.pte; done; done
ln -sfn $M/tokenizer.model tokenizer.model; ls -l | head -12
sha256sum $GROOT/sarc-tools/verify.sh; chmod +x $GROOT/tools/*.sh $GROOT/tools/*.py $GROOT/sarc-tools/verify.sh" | tee -a $A/logs/sync.log
echo "local verify.sh: $(sha256sum "$ET/sarc/tools/verify.sh")" | tee -a $A/logs/sync.log
