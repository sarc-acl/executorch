#!/bin/bash
# redispatch.sh <build tag> <session...>: collect the dispatched names again for sessions whose collection stopped
# early (cool_start's status under set -e, fixed in host.sh); the partial collection is kept under superseded/.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"
TAG=${1:?build tag}; shift
for name in "$@"; do
  S=$A/stage/$name
  if [[ -e $S/dispatch && ! -e $S/dispatch/dispatch.done ]]; then
    mkdir -p "$A/superseded/dispatch-cool-start-status/$name"
    mv "$S/dispatch" "$S/dispatch.out" "$A/superseded/dispatch-cool-start-status/$name/"
  fi
  [[ -e $S/dispatch/dispatch.done ]] && continue
  "$TOOLS/matrix_dispatch.sh" "$name" "$TAG" > "$S/dispatch.out" 2>&1 || { echo "dispatch $name failed" >&2; exit 1; }
done
