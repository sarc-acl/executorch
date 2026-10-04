#!/bin/bash
# gl.sh <command...>: run one GPU job under the Xe2 (B70) gpu-lab lock (one GPU job at a time). Refuses to start
# while a GPU process this campaign did not start is running and fails (76) if one appears during the job.
. "$(dirname "$(readlink -f "$0")")/host.sh"
exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK"; flock -w 1800 9 || { echo "gpu-lab lock busy"; exit 75; }
F=$(mktemp); guarded $F "$@" 9>&-; rc=$?; [[ -s $F ]] && cat $F >&2; rm -f $F; exit $rc
