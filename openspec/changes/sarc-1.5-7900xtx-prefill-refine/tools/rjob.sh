#!/bin/bash
# rjob.sh <name> <command...>: runs ON THE GPU HOST, started detached by rstart (rlib.sh). Keeps <gpu-root>/logs/<name>.status:
# "RUNNING <utc>" while it runs, "DONE rc=<rc> <utc>" after. Refuses a name that is RUNNING (never a second copy of a session).
source "$(dirname "$(readlink -f "$0")")/env.sh"
n=$1; shift; f=$A/logs/$n.status; mkdir -p $A/logs
grep -qs '^RUNNING' $f && { echo "$n is RUNNING" >&2; exit 3; }
echo "RUNNING $(date -u +%FT%TZ)" > $f
"$@"; rc=$?
echo "DONE rc=$rc $(date -u +%FT%TZ)" >> $f; exit $rc
