#!/bin/bash
# session.sh <session> [e2e5 args]: wait for the GPU to cool (<= 50 C or 5 min), then run e2e5.sh on stage/<session>.
source "$(dirname "$0")/common.sh"; S=$1; shift
cool_start 50 300
"$(dirname "$0")/e2e5.sh" --stage $A/stage/$S --out raw --lock $LOCK "$@"
