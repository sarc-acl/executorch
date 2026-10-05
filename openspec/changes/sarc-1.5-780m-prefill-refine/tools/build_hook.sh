#!/bin/bash
# build_hook.sh <artifact dir> <tag> [build.sh options, default --llama; --tests-only for none]: build the working copy plus the
# release-zone hook patches of ../hooks/ in a scratch tree (<artifact dir>/hook-tree/executorch); the working copy
# itself never carries the hooks. Output <artifact dir>/build/<tag>, source state in build/<tag>.src.txt.
# CPU only; holds <artifact dir>/PAUSE so the sweep waits meanwhile.
set -euo pipefail
A=$(realpath "$1"); TAG=$2; shift 2; OPTS=("${@:---llama}"); [[ ${OPTS[0]} == --tests-only ]] && OPTS=()
ET=$(cd "$(dirname "$0")/../../../.." && pwd); C=$ET/openspec/changes/sarc-1.5-780m-prefill-refine
S=$A/hook-tree/executorch; mkdir -p "$S" "$A/logs"
touch "$A/PAUSE"; trap 'rm -f "$A/PAUSE"' EXIT
rsync -a --delete --exclude .git --exclude __pycache__ "$ET/" "$S/"
for p in "$C"/hooks/*.patch; do patch -s -p1 -d "$S" < "$p"; done
{ echo "tag $TAG $(date -u +%FT%TZ) options ${OPTS[*]}"; git -C "$ET" rev-parse HEAD; git -C "$ET" status --short
  sha256sum "$C"/hooks/*.patch; } > "$A/build/$TAG.src.txt" 2>/dev/null || { mkdir -p "$A/build"; git -C "$ET" rev-parse HEAD > "$A/build/$TAG.src.txt"; }
SARC_MOUNT_ROOT=$(dirname "$ET") "$ET/sarc/tools/build.sh" "${OPTS[@]}" "$S" "$A/build/$TAG" > "$A/logs/build-$TAG.log" 2>&1 \
  || { echo "$TAG: BUILD FAILED, see logs/build-$TAG.log"; exit 1; }
echo "$TAG: built $(date -u +%FT%TZ)"
