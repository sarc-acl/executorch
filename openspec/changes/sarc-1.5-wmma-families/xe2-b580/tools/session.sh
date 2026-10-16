#!/bin/bash
# session.sh <session> [e2e5 args]: wait for the card to cool (host.sh cool_start), then run e2e5.sh on
# stage/<session>. The first session of the campaign (baseline, A/A) passes --calibrate; every later one uses
# the clock threshold and idle temperature that session recorded. Exit status = e2e5.sh's (76 = aborted by a foreign GPU process).
. "$(dirname "$(readlink -f "$0")")/host.sh"
if [[ ${WMMA_UNIT:-} != "$0" ]]; then
  exec "$TOOLS/hold.sh" run "session.sh ${1:-}" env WMMA_UNIT="$0" "$0" "$@"
fi

S=$1; shift
busy_wait "session $S" || exit $?   # before the build lock: a build may run while the session waits
gpu_shared || exit 75; cool_start
$TOOLS/e2e5.sh --stage $A/stage/$S --out raw --lock $LOCK "$@"; exit $?
