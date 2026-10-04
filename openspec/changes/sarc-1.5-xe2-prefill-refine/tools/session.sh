#!/bin/bash
# session.sh <session> [e2e5 args]: wait for the GPU to cool (<= 48 C or 5 min), then run e2e5.sh on stage/<session>.
. "$(dirname "$(readlink -f "$0")")/host.sh"; S=$1; shift
cool_start
$TOOLS/e2e5.sh --stage $A/stage/$S --out raw --lock $LOCK "$@"
