#!/bin/bash
# drun.sh <job name> <tool> [args...]: workstation side. Starts one tool of this directory on the device, detached
# (setsid nohup), so it survives the loss of this workstation or of the ssh session. On the device:
#   ~/hmz-sarc-orin/jobs/<job>.out   its output
#   ~/hmz-sarc-orin/jobs/<job>.status "RUNNING <utc> <command>" then "DONE rc=<rc> <utc>"
# jobs/<job>.pid is the session leader of the job (dkill.sh ends the whole job by it, never by a name pattern).
# A job name can be used once. dstat.sh shows the jobs; pull.sh copies the results back.
source "$(dirname "$0")/common.sh"; [[ $SIDE == ws ]] || { echo "workstation only" >&2; exit 2; }
J=$1; shift; T=$DEVROOT/executorch/$CHANGE_REL/tools
Q=$(printf '%q ' "$@")
dssh "cd $DEVROOT && test ! -e jobs/$J.status || { echo 'job $J exists' >&2; exit 2; }
echo \"RUNNING \$(date -u +%FT%TZ) $Q\" > jobs/$J.status
setsid nohup bash -c 'cd ~/$T && ./$Q > ~/$DEVROOT/jobs/$J.out 2>&1; echo \"DONE rc=\$? \$(date -u +%FT%TZ)\" >> ~/$DEVROOT/jobs/$J.status' > /dev/null 2>&1 < /dev/null &
echo \$! > jobs/$J.pid; sleep 1; cat jobs/$J.status"
