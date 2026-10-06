#!/bin/bash
# build_tag.sh <tag> <commit>: one build tag = one exported commit, built once (R5).
#   MB_ONLY=1: backend and tests only (test_llama_microbench), no llama_main: for kernel screens, never timed e2e.
#   src/m51/<tag>/   export of <commit> with its submodules (export_commit.sh) + MANIFEST.txt
#   build/m51/<tag>/ android build (build-native.sh --llama: backend, test_llama_microbench, test_sarc_select,
#                    llama_main) and build/m51/<tag>-etdump/ (llama_main with ETDump)
#   build/m51/<tag>.txt  commit, manifest digest, toolchain, start/end, rc, golden check output
# Builds run at nice 19 with J=8 (another campaign measures on this host), wait while a timed session runs
# (dev.sh other_timed_session), and are suspended while one runs after they started. One coordinator-hold unit.
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/dev.sh"
TAG=$1; C=$(git -C "$TOOLS" rev-parse "$2^{commit}") || exit 2
SRC=$ART/src/m51/$TAG; B=$ART/build/m51/$TAG; R=$ART/build/m51/$TAG.txt
[[ -e $B || -e $SRC ]] && { echo "tag $TAG exists: a build tag is built once"; exit 1; }
while other_timed_session; do sleep 60; done
exec 3>>"$R"
{ echo "tag $TAG commit $C${MB_ONLY:+ (MB_ONLY: no llama_main)}"; echo "start $(date -u +%FT%TZ)"; } >&3
# The build runs in its own process group, suspended (SIGSTOP) while a timed session runs and resumed after it:
# a session can start after the build has started.
setsid "$TOOLS/hold.sh" run "build $TAG" bash -c '
  set -e; "$1" "$2" "$3" >&4
  ln -s "$5" "$3/executorch/.venv"
  if [[ -n $7 ]]; then OUT_DIR=$4 nice -n 19 "$6/build-native.sh" "$3/executorch" android; exit; fi
  OUT_DIR=$4 nice -n 19 "$6/build-native.sh" "$3/executorch" android --llama
  OUT_DIR=$4-etdump nice -n 19 "$6/build-native.sh" "$3/executorch" android --etdump --no-tests
' _ "$TOOLS/export_commit.sh" "$C" "$SRC" "$B" "$ART/venv/m51" "$TOOLS" "${MB_ONLY:-}" > "$ART/build/m51/$TAG.log" 2>&1 4>&3 &
bp=$!; stopped=0
while kill -0 $bp 2>/dev/null; do
  if other_timed_session; then
    (( stopped )) || { kill -STOP -- -$bp 2>/dev/null; stopped=1; echo "suspended $(date -u +%FT%TZ) (timed session)" >&3; }
  elif (( stopped )); then kill -CONT -- -$bp 2>/dev/null; stopped=0; echo "resumed $(date -u +%FT%TZ)" >&3; fi
  sleep 20
done
wait $bp; rc=$?
{ echo "rc $rc"; echo "manifest_sha256 $(sha256sum "$SRC/MANIFEST.txt" | cut -d' ' -f1)"
  echo "ndk $(grep -m1 Pkg.Revision "${NDK:-<android-ndk>}/source.properties")"
  echo "glslc $("${GLSLC:-<vulkan-sdk>/glslc}" --version | head -1)"
  echo "golden (working-copy sarc/tools/spirv_golden.py; native glslc, so a mismatch is expected: pending):"
  "$ART/venv/m51/bin/python" "$TOOLS/../../../../sarc/tools/spirv_golden.py" "$B/vulkan_compute_shaders" "$SRC/executorch/sarc/golden/spirv.json" 2>&1
  echo "end $(date -u +%FT%TZ)"; } >&3
echo "BUILD $TAG rc=$rc"; cat "$R"
