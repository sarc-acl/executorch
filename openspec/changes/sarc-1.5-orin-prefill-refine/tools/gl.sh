#!/bin/bash
# gl.sh <command...>: run one GPU job under the Orin gpu-lab lock (one GPU job at a time). Device side. Refuses to start
# when a GPU process this campaign did not start is present or the GPU sensors do not answer. While the job runs it looks for foreign GPU processes every 0.5 s and once more at the end (exit 76 on any
# sighting, from what was captured: the job's result is not usable); it also checks the card again (exit 70);
# otherwise it returns the job's exit status.
source "$(dirname "$0")/common.sh"
[[ $SIDE == device ]] || { echo "device only" >&2; exit 2; }
take_lock; gone_check "gl.sh before $(basename "$1")"; no_others "gl.sh $(basename "$1")"
W=$(mktemp); others_watch_start $W
"$@" 9>&-; rc=$?
oth=$(others_watch_stop $W); rm -f $W
gone_check "gl.sh after $(basename "$1") rc=$rc"
[[ -n $oth ]] && abort_others "gl.sh during $(basename "$1") rc=$rc" "$oth"
exit $rc
