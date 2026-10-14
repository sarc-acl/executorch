#!/bin/bash
# build-extra.sh <tag>: workstation side. Adds logits_dump (tools/jetson-cross/logits_dump, the sibling campaign's
# tool unchanged) to the bundle of an existing build tag, under the desktop build lock; hash appended to
# build/<tag>.src.txt. Deploy afterwards with deploy.sh --extra <tag>.
set -uo pipefail
source "$(dirname "$0")/common.sh"; TAG=$1; W=$A/build/$TAG; need $A/build/$TAG.src.txt $W/bundle/llama_main
[[ -e $W/bundle/logits_dump ]] && { echo "logits_dump exists for $TAG" >&2; exit 2; }
flock ~/.cache/gpu-lab/lock-desktop-build podman run --rm --userns=keep-id -v "$TOOLS/jetson-cross:/recipe:ro,Z" -v "$W:/work:Z" \
  localhost/et-jetson-cross:jp7.2.1 bash /recipe/extra.sh > $A/build/$TAG.extra.log 2>&1; rc=$?
[[ $rc == 0 ]] && sha256sum $W/bundle/logits_dump >> $A/build/$TAG.src.txt
echo "BUILD_EXTRA_DONE $TAG rc=$rc"; exit $rc
