#!/usr/bin/env bash
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
work=$(realpath "${JETSON_CROSS_WORK:-$recipe/../../out/jetson-cross}")
exec podman run --rm --userns=keep-id \
  -v "$recipe:/recipe:ro,Z" -v "$work:/work:Z" \
  localhost/et-jetson-cross:jp7.2.1 bash /recipe/build.sh "$@"
