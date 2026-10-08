#!/bin/bash
# session.sh <session> (--calibrate | --clkmin-file JSON) [e2e5 args]: wait for the GPU to cool (<= 50 C or 5 min),
# then run e2e5.sh on stage/<session>. --calibrate (record-only clock) is for the baseline and A/A sessions.
source "$(dirname "$0")/common.sh"; S=$1; shift
cool_start 50 300
"$TOOLS/e2e5.sh" --stage $A/stage/$S --out raw --lock $LOCK "$@"
