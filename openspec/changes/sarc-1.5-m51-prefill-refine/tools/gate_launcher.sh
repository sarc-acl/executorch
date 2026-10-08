#!/bin/bash
# gate_launcher.sh <compiler command...>: compiler launcher of this campaign's builds (CMAKE_*_COMPILER_LAUNCHER, in front of ccache).
# A build must be quiet while a timed session runs (RULES R5, on this workstation also the other campaign's). SIGSTOP cannot be used for that
# here: in the agent's tool environment `kill -STOP` is reported by the shell as "Stopped" but the process keeps running (a counting
# loop kept counting, /proc showed S; checked 2026-10-08), so a "suspended" line in a build record proved nothing. Instead every compiler
# invocation first waits while a timed session runs (dev.sh other_timed_session); invocations already running finish (seconds), then no
# new one starts until the session is over. Not gated: the shader code generation and glslc of the vulkan backend (custom commands, not
# compiler launches). A wait is logged to $GATE_LOG (the build record) once per invocation.
source "$(dirname "$(readlink -f "$0")")/dev.sh"
if other_timed_session; then
  t0=$SECONDS
  while other_timed_session; do sleep 5; done
  [[ -n ${GATE_LOG:-} ]] && echo "gated $(date -u +%FT%TZ) waited $((SECONDS - t0)) s before: $(basename "$1")" >> "$GATE_LOG"
fi
exec "$@"
