#!/bin/bash
# gl.sh <command...>: run one GPU job under the 4070 Ti gpu-lab lock (one GPU job at a time). Refuses to start
# when a GPU process this campaign did not start is present or the card is gone; checks the card again after the
# job (exit 70 on loss, see common.sh) and otherwise returns the job's exit status.
source "$(dirname "$0")/common.sh"
take_lock; gone_check "gl.sh before $(basename "$1")"; no_others "gl.sh $(basename "$1")"
"$@" 9>&-; rc=$?
gone_check "gl.sh after $(basename "$1") rc=$rc"
exit $rc
