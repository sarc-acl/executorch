#!/bin/bash
# wsrun.sh <job name> <tool> [args...]: workstation side. Runs one tool of this directory detached (setsid nohup)
# from a private copy of the tools (jobs/<job>.tools/), so that editing the working copy while a long job runs
# cannot change the script under it. jobs/<job>.out = output, jobs/<job>.status = "RUNNING ..." then "DONE rc=..".
# These jobs do not survive a reboot of the workstation: STATUS.md must say what is running.
source "$(dirname "$0")/common.sh"; [[ $SIDE == ws ]] || { echo "workstation only" >&2; exit 2; }
J=$1; shift; mkdir -p $A/jobs; [[ -e $A/jobs/$J.status ]] && { echo "job $J exists" >&2; exit 2; }
rsync -a --exclude __pycache__ $TOOLS/ $A/jobs/$J.tools/
echo "RUNNING $(date -u +%FT%TZ) $*" > $A/jobs/$J.status
setsid nohup bash -c "cd $A/jobs/$J.tools && $(printf '%q ' "$@") > $A/jobs/$J.out 2>&1; echo \"DONE rc=\$? \$(date -u +%FT%TZ)\" >> $A/jobs/$J.status" > /dev/null 2>&1 < /dev/null &
cat $A/jobs/$J.status
