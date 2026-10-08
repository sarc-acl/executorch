# rlib.sh: helpers of the control workstation for the GPU host (sourced after env.sh). Every command sent to the GPU host is appended
# to the device command log of the workspace (instruction-for-ai/README.md rule 5).
# GPUHOST (ssh alias), GROOT (<gpu-root>) come from $A/env.local.
CMDLOG=${CMDLOG_DIR:-$(cd "$A/../.." && pwd)/new-workspace/.artifacts}/cmd-log-$(date -u +%F).sh
clog() { mkdir -p "$(dirname "$CMDLOG")"; echo "# $(date -u +%FT%TZ) campaign 7900xtx: $*" >> "$CMDLOG"; }
# rsh <bash script text on stdin or $1>: run a short command on the GPU host through a bash -s (never the login shell)
rsh() { clog "ssh <gpu-host> bash -s: ${1:-<stdin script>}" ; if [[ $# -gt 0 ]]; then ssh -o BatchMode=yes -o ConnectTimeout=20 "$GPUHOST" 'bash -s' <<< "$1"
  else ssh -o BatchMode=yes -o ConnectTimeout=20 "$GPUHOST" 'bash -s'; fi; }
# rstart <name> <command...>: start a detached job on the GPU host through tools/rjob.sh (stdin/stdout detached, L40); log
# $GROOT/logs/<name>.log, status file $GROOT/logs/<name>.status ("RUNNING <utc>", then "DONE rc=<rc> <utc>"). The command path is
# relative to $GROOT. Refuses to start a name that is RUNNING; never starts a second copy.
rstart() { local n=$1; shift; local q; q=$(printf '%q ' "$@"); clog "ssh -n -f <gpu-host> setsid nohup rjob.sh $n: $q"
  ssh -o BatchMode=yes -n -f "$GPUHOST" "cd $GROOT && setsid nohup $GROOT/tools/rjob.sh $n $q > $GROOT/logs/$n.log 2>&1 < /dev/null &"; }
rstatus() { rsh "cat $GROOT/logs/$1.status 2>&1; tail -n ${2:-5} $GROOT/logs/$1.log 2>&1"; }
