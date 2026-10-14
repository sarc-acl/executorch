#!/bin/bash
# gl.sh <command...>: run one GPU job under the RX 7900 XTX gpu-lab lock (GPU host) (one GPU job at a time), as one unit of the
# coordinator hold (hold.sh; HOLD is looked at again once both locks are held). Refuses to start while a foreign
# GPU user is present (others.sh). Host builds (this campaign's or others') are not waited for: gl.sh jobs are
# correctness checks or GPU-timestamp kernel timings, not timed sessions (R5); the load is logged. Profiler tracing variables (MESA_VK_TRACE*,
# RADV_THREAD_TRACE*) are refused here: a capture runs only through its own detached job (rule R9).
source "$(dirname "$(readlink -f "$0")")/env.sh"
if env | grep -q -E "^(MESA_VK_TRACE|RADV_THREAD_TRACE)"; then echo "gl.sh: REFUSED: tracing variable set" >&2; exit 97; fi
H=$T/hold.sh; eval "$("$H" vars)"; what="gl.sh $*"
while :; do
  "$H" wait "${what:0:240}"; exec 8>>"$BUSY"; flock -s 8
  exec 9>>"$LOCKF"; flock -w 3600 9 || { echo "gpu-lab lock busy"; exit 75; }
  [[ -e $HOLD ]] || break
  exec 9>&- 8>&-
done
# a foreign GPU user (a monitor of the owner, another user's job): wait for it to go (at most 2 h), never force; the wait is logged
t0=$SECONDS; while :; do o=$($T/others.sh); [[ $o == "gpu= "* ]] && break
  (( SECONDS - t0 >= 7200 )) && { echo "gl.sh: foreign GPU user for 2 h: $o" >&2; exit 76; }; sleep 10; done
(( SECONDS - t0 > 0 )) && echo "$(date -u +%FT%TZ) gl.sh waited $((SECONDS - t0)) s for a foreign GPU user" >> $A/logs/gl.log
echo "$(date -u +%FT%TZ) gl.sh load=$(cut -d" " -f1 /proc/loadavg) $o $*" >> $A/logs/gl.log
"$@" 9>&-
