#!/bin/bash
# buildq.sh <job to wait for> <tag> <commit> [base tag]: workstation side. Waits for an earlier workstation job
# (wsrun.sh) to end, then runs build-orin.sh <tag> <commit>, with BASE_TAG=<base tag> when given (tree = hard links
# to that tag's tree plus the changed files). One cross build at a time, each under the desktop build lock.
source "$(dirname "$0")/common.sh"; [[ $SIDE == ws ]] || { echo "workstation only" >&2; exit 2; }
while ! grep -q '^DONE' $A/jobs/$1.status 2>/dev/null; do sleep 20; done
grep -q '^DONE rc=0' $A/jobs/$1.status || { echo "job $1 failed: $(tail -1 $A/jobs/$1.status)" >&2; exit 3; }
BASE_TAG=${4:-} exec "$TOOLS/build-orin.sh" "$2" "$3"
