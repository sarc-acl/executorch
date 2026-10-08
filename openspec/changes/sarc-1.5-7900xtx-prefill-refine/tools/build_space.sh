#!/bin/bash
# build_space.sh <artifact dir> <batch...>: one test_llama_microbench per sweep batch of plan_space.py.
# The variants of a batch are generated into a scratch copy of the working copy (<artifact dir>/sweep-tree/executorch,
# never committed), built with the unmodified sarc/tools/build.sh into one reused build directory, and the binary is
# kept as <artifact dir>/bin/microbench-<batch>. CPU only; holds <artifact dir>/PAUSE so the sweep waits meanwhile.
set -euo pipefail
A=$(realpath "$1"); shift
ET=$(cd "$(dirname "$0")/../../../.." && pwd); T=$ET/openspec/changes/sarc-1.5-780m-prefill-refine/tools
S=$A/sweep-tree/executorch; mkdir -p "$S" "$A/bin" "$A/logs"
touch "$A/PAUSE"; trap 'rm -f "$A/PAUSE"' EXIT
for b in "$@"; do
  [[ -x $A/bin/microbench-$b ]] && { echo "$b: already built"; continue; }
  rsync -a --delete --exclude .git --exclude __pycache__ "$ET/" "$S/"
  python3 "$T/gen_space.py" "$S" "$A/space/${SPACE_PLAN:-sweep}/$b" > "$A/logs/gen-$b.log"
  SARC_MOUNT_ROOT=$(dirname "$ET") "$(dirname "$(readlink -f "$0")")/hold.sh" run "build batch $b" "$ET/sarc/tools/build.sh" "$S" "$A/build/space" > "$A/logs/build-$b.log" 2>&1 \
    || { echo "$b: BUILD FAILED, see logs/build-$b.log"; exit 1; }
  cp -f "$A/build/space/tests/test_llama_microbench" "$A/bin/microbench-$b"
  echo "$b: built $(tail -1 "$A/logs/gen-$b.log") $(date -u +%FT%TZ)"
done
