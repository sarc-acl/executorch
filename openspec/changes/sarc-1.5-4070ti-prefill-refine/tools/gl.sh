#!/bin/bash
# gl.sh <command...>: run one GPU job under the 4070 Ti gpu-lab lock (one GPU job at a time), refusing to start
# while a GPU process we did not start is running or when the card is gone.
source "$(dirname "$0")/common.sh"
exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK"; flock -w 1800 9 || { echo "gpu-lab lock busy"; exit 75; }
gpu_alive || gpu_gone gl.sh
o=$(others); [[ -n $o ]] && { echo "other GPU process: $o"; exit 76; }
"$@" 9>&-
