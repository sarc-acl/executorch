#!/bin/bash
# dkill.sh <job> <reason>: workstation side. Ends a device job started with drun.sh: TERM to its whole session
# (the pid in jobs/<job>.pid is the session leader), and the reason is appended to its status file.
source "$(dirname "$0")/common.sh"; J=$1; shift
dssh "cd $DEVROOT && p=\$(cat jobs/$J.pid) && pkill -TERM -s \$p; sleep 1; pgrep -s \$p -a | cut -c1-80; echo \"KILLED \$(date -u +%FT%TZ) $*\" >> jobs/$J.status; tail -1 jobs/$J.status"
