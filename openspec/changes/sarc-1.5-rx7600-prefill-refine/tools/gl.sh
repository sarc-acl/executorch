#!/bin/bash
# gl.sh <command...>: run one GPU job under the RX 7600 gpu-lab lock (one GPU job at a time), as one unit of the
# coordinator hold (hold.sh; HOLD is looked at again once both locks are held). Refuses to start while a foreign
# GPU user is present (others.sh); waits out this campaign's own builds. Other host builds are not waited for: gl.sh
# jobs are correctness checks or GPU-timestamp kernel timings, not timed sessions (R5); the load is logged. Profiler tracing variables (MESA_VK_TRACE*,
# RADV_THREAD_TRACE*) are refused here: a capture runs only through its own detached job (rule R9).
source "$(dirname "$(readlink -f "$0")")/env.sh"
if env | grep -q -E "^(MESA_VK_TRACE|RADV_THREAD_TRACE)"; then echo "gl.sh: REFUSED: tracing variable set" >&2; exit 97; fi
H=$T/hold.sh; eval "$("$H" vars)"; what="gl.sh $*"
while [[ -e $A/.building ]]; do sleep 20; done
while :; do
  "$H" wait "${what:0:240}"; exec 8>>"$BUSY"; flock -s 8
  exec 9>>"$LOCKF"; flock -w 3600 9 || { echo "gpu-lab lock busy"; exit 75; }
  [[ -e $HOLD ]] || break
  exec 9>&- 8>&-
done
o=$($T/others.sh); [[ $o == "gpu= "* ]] || { echo "gl.sh: foreign GPU user: $o" >&2; exit 76; }
echo "$(date -u +%FT%TZ) gl.sh load=$(cut -d" " -f1 /proc/loadavg) $o $*" >> $A/logs/gl.log
"$@" 9>&-
