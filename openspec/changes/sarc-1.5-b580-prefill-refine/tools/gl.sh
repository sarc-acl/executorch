#!/bin/bash
# gl.sh <command...>: run one GPU job under the B580 gpu-lab lock (one GPU job at a time) and the desktop-build
# lock (shared: no build during it). Refuses to start while a GPU workload this campaign did not start is running
# and fails (76) if one appears during the job. Desktop DRM clients do not stop it (host.sh).
. "$(dirname "$(readlink -f "$0")")/host.sh"
gpu_shared || exit 75
exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK"; flock -w 1800 9 || { echo "gpu-lab lock busy"; exit 75; }
F=$(mktemp); guarded $F "$@" 9>&-; rc=$?; [[ -s $F ]] && cat $F >&2; rm -f $F; exit $rc
