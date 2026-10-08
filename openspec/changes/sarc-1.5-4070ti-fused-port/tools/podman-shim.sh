#!/bin/bash
# podman -> docker shim for sarc/tools/build.sh (this host has docker only). Install as $A/bin/podman.
# `--userns=keep-id` (podman only) becomes `--user uid:gid` with a writable HOME.
args=()
for a in "$@"; do
  if [[ $a == --userns=keep-id ]]; then args+=(--user "$(id -u):$(id -g)" -e HOME=/tmp); else args+=("$a"); fi
done
exec docker "${args[@]}"
