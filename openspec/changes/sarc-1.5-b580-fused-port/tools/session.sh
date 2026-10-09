#!/bin/bash
# session.sh <session> [e2e5 args]: wait for the card to cool (host.sh cool_start), then run e2e5.sh on
# stage/<session>. The first session of the campaign (baseline, A/A) passes --calibrate; every later one uses
# the clock threshold and idle temperature that session recorded. Exit status = e2e5.sh's (76 = aborted by a foreign GPU process).
. "$(dirname "$(readlink -f "$0")")/host.sh"; S=$1; shift
hold_wait "session $S"; busy_wait "session $S"; idle_wait "session $S"   # before the build lock: a build may run while the session waits for an idle desktop
gpu_shared || exit 75; cool_start
# --extra 12: with the idle wait lifted (owner decision 2026-10-09) the desktop disturbs more runs; a run above
# BUSYMAX is invalid and replaced as before, by up to 12 extra interleaved pairs a cell instead of e2e5.sh's 3.
$TOOLS/e2e5.sh --stage $A/stage/$S --out raw --lock $LOCK --extra 12 "$@"; exit $?
