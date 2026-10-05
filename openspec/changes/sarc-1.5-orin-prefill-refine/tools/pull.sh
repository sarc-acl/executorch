#!/bin/bash
# pull.sh: workstation side. Mirrors the device's results into <artifacts>/device/ (stage/, raw/, probe/, jobs/,
# markers), without the binaries. The device copy stays where it is; nothing is deleted on either side.
source "$(dirname "$0")/common.sh"; [[ $SIDE == ws ]] || { echo "workstation only" >&2; exit 2; }
mkdir -p $A/device
rsync -a --exclude llama_main --exclude 'libllama_runner.so' --exclude test_llama_microbench --exclude logits_dump --exclude 'verify-bin' \
  --exclude '/build' --exclude '/executorch' --exclude '*.bin' ${PULL_EXTRA:-} $DEVICE:$DEVROOT/ $A/device/
du -sh $A/device | cut -f1
