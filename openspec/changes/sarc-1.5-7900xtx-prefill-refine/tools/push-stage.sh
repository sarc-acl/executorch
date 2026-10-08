#!/bin/bash
# push-stage.sh <session>: copy stage/<session> (binaries, env, prompts) to the GPU host; pull-stage.sh <session>: bring the raw runs and logs back.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"; source $T/rlib.sh
S=$1; [[ -d $A/stage/$S ]] || { echo "no stage/$S" >&2; exit 2; }
case $(basename "$0") in
  push-stage.sh) clog "rsync stage/$S -> <gpu-root>/stage/$S"
    rsync -a --info=stats1 -e "ssh -o BatchMode=yes" "$A/stage/$S/" "$GPUHOST:$GROOT/stage/$S/" | grep -E 'Number of regular|Total transferred' ;;
  pull-stage.sh) clog "rsync <gpu-root>/stage/$S -> stage/$S (results only)"
    rsync -a --info=stats1 -e "ssh -o BatchMode=yes" --exclude=llama_main --exclude=libllama_runner.so --exclude=test_llama_microbench --exclude=lp \
      "$GPUHOST:$GROOT/stage/$S/" "$A/stage/$S/" | grep -E 'Number of regular|Total transferred' ;;
esac
