#!/bin/bash
# extraq.sh <job to wait for> <tag> [...]: workstation side. Waits for an earlier workstation job, then adds
# logits_dump to each build tag (build-extra.sh, under the desktop build lock).
source "$(dirname "$0")/common.sh"; [[ $SIDE == ws ]] || { echo "workstation only" >&2; exit 2; }
while ! grep -q '^DONE' $A/jobs/$1.status 2>/dev/null; do sleep 20; done; shift
for t in "$@"; do hold_wait; "$TOOLS/build-extra.sh" $t || exit 3; done
