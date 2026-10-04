#!/bin/bash
# gl.sh <command...>: run one GPU job under the Xe2 (B70) gpu-lab lock (one GPU job at a time), refusing to start
# while another known GPU process is running.
. "$(dirname "$(readlink -f "$0")")/host.sh"
exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK"; flock -w 1800 9 || { echo "gpu-lab lock busy"; exit 75; }
o=$(gpu_others)
[[ -n $o ]] && { echo "other GPU process: $o"; exit 76; }

"$@" 9>&-
