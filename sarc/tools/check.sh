#!/bin/bash
# GPU-free checks for dev/1.5 (any agent can run them on a build host):
#   1. zone rule: outside the release and dev zones only sarc/HOOKS paths differ
#      from release/1.5 (committed, staged, unstaged and untracked changes)
#   2. twin template wrappers are identical from #version on
#   3. test_sarc_select (release tables alone, and with the dev zone +
#      ET_VK_SARC_UNVERIFIED=1)
#   4. the release export builds (host; --android adds the Android build)
#   5. shipped SPIR-V matches sarc/golden/spirv.json
# usage: check.sh [--no-build] [--android] [--work <dir>]
#   --work: scratch for the export and builds (default <workspace>/.artifacts/sarc-1.5/check)
set -uo pipefail
SARC_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
. "$SARC_ROOT/sarc/tools/zones.sh"
cd "$SARC_ROOT"
BUILD=1; ANDROID=0; WORK=$(realpath -m "$SARC_ROOT/../../../.artifacts/sarc-1.5/check")
while [[ $# -gt 0 ]]; do
  case $1 in --no-build) BUILD=0 ;; --android) ANDROID=1 ;; --work) WORK=$(realpath -m "$2"); shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;; esac; shift
done
FAIL=0; step() { echo "== $*"; }; bad() { echo "FAIL: $*"; FAIL=1; }

step "1 zone rule vs $SARC_BASE"
mapfile -t HOOKS < <(sarc_hooks)
while read -r p; do
  [[ -z $p ]] && continue
  sarc_in_zone "$p" "${SARC_RELEASE_ZONE[@]}" "${SARC_DEV_ZONE[@]}" && continue
  printf '%s\n' "${HOOKS[@]}" | grep -qx -- "$p" && continue
  bad "$p is outside the zones and not in sarc/HOOKS"
done < <({ git diff --name-only "$SARC_BASE"; git ls-files --others --exclude-standard; } | sort -u)
for h in "${HOOKS[@]}"; do
  git diff --quiet "$SARC_BASE" -- "$h" && echo "note: HOOKS entry $h is unchanged (stale?)"
done

step "2 twin wrappers"
for t in "${SARC_TWINS[@]}"; do
  set -- $t
  cmp -s <(sed -n '/^#version/,$p' "$1") <(sed -n '/^#version/,$p' "$2") || bad "twins differ: $1 $2"
done

step "3 test_sarc_select"
I=backends/vulkan/runtime/graph/ops/impl; T=$(mktemp -d)
YAMLS=$(ls backends/vulkan/runtime/graph/ops/glsl/sarc/*.yaml backends/vulkan/runtime/graph/ops/glsl/sarc_dev/*.yaml)
CXX=${CXX:-c++}
# Only the selection sources are GPU-free: Select.cpp and table_*.cpp.
SEL=$(ls $I/sarc/Select.cpp $I/sarc/table_*.cpp)
$CXX -std=c++17 -Wall -I"$SARC_ROOT/.." backends/vulkan/test/sarc_dev/test_sarc_select.cpp $SEL -o "$T/rel" \
  && "$T/rel" $YAMLS || bad "test_sarc_select (release tables)"
$CXX -std=c++17 -Wall -I"$SARC_ROOT/.." backends/vulkan/test/sarc_dev/test_sarc_select.cpp $SEL \
  $I/sarc_dev/*.cpp -o "$T/dev" && ET_VK_SARC_UNVERIFIED=1 "$T/dev" $YAMLS || bad "test_sarc_select (dev zone)"
rm -rf "$T"

if [[ $BUILD == 1 ]]; then
  step "4 release export build ($WORK)"
  "$SARC_ROOT/sarc/tools/make-release.sh" --export "$WORK/export/executorch" >/dev/null || bad "export"
  if "$SARC_ROOT/sarc/tools/build.sh" "$WORK/export/executorch" "$WORK/build" > "$WORK/build-host.log" 2>&1; then
    echo "host build OK"
  else bad "host build of the release export (see $WORK/build-host.log)"; fi
  if [[ $ANDROID == 1 ]]; then
    "$SARC_ROOT/sarc/tools/build.sh" --android "$WORK/export/executorch" "$WORK/build" > "$WORK/build-android.log" 2>&1 \
      && echo "android build OK" || bad "android build of the release export (see $WORK/build-android.log)"
  fi
  step "5 SPIR-V golden"
  python3 "$SARC_ROOT/sarc/tools/spirv_golden.py" "$WORK/build/backend/vulkan_compute_shaders" \
    "$SARC_ROOT/sarc/golden/spirv.json" || bad "spirv golden"
fi
echo "check.sh: $([[ $FAIL == 0 ]] && echo PASS || echo FAIL)"
exit $FAIL
