#!/bin/bash
# deploy.sh DEVICE NAME...: creates the device's one fresh directory on first use (recorded in
# OUT/DEVICE/remote-directory.txt), copies the runner, its Vulkan runtime library, run_models.sh and
# models/NAME.pte + models/MODEL-input.bin there, and appends the sha256 computed on the device to
# OUT/DEVICE/deployed.txt. B580 (local) gets OUT/b580/stage with symlinks instead.
set -eu
. "$(dirname "$0")/devices.sh"; device_info "$1"; DEV=$1; shift
mkdir -p "$OUT/$DEV/results"
if [ $KIND = local ]; then
  mkdir -p "$RDIR"; cd "$RDIR"
  ln -sfn "$RUNNER_BIN" executor_runner; ln -sfn "$RUNNER_LIB" libexecutorch_vulkan_shared_runtime.so
  ln -sfn "$OUT/tools/run_models.sh" run_models.sh; ln -sfn "$OUT/models" models; ln -sfn "$OUT/$DEV/results" results
  sha256sum -L executor_runner libexecutorch_vulkan_shared_runtime.so run_models.sh >> "$OUT/$DEV/deployed.txt" 2>/dev/null \
    || sha256sum "$RUNNER_BIN" "$RUNNER_LIB" "$OUT/tools/run_models.sh" >> "$OUT/$DEV/deployed.txt"
  exit 0
fi
if [ -z "$RDIR" ]; then
  case $KIND in
    ssh) RDIR=$(rsh "mktemp -d /tmp/vk-model-survey.XXXXXXXX");;
    adb) RDIR=/data/local/tmp/vk-model-survey.$(head -c6 /dev/urandom | od -An -tx1 | tr -d ' \n')
         rsh "test ! -e $RDIR && mkdir $RDIR" ;;
  esac
  printf '%s\n' "$RDIR" > "$OUT/$DEV/remote-directory.txt"
  echo "created $RDIR on $DEV $(date -u +%FT%TZ)" >> "$OUT/$DEV/remote-directory.txt"
fi
files=("$RUNNER_BIN" "$RUNNER_LIB" "$OUT/tools/run_models.sh")
models=()
for n in "$@"; do models+=("$OUT/models/$n.pte" "$OUT/models/${n%%-*}-input.bin"); done
mapfile -t models < <(printf '%s\n' "${models[@]}" | sort -u)
case $KIND in
  ssh) scp -q "${files[@]}" "$HOST:$RDIR/"; rsh "mkdir -p $RDIR/models $RDIR/results"
       [ ${#models[@]} -gt 0 ] && scp -q "${models[@]}" "$HOST:$RDIR/models/";;
  adb) adb -s "$SERIAL" push "${files[@]}" "$RDIR/" > /dev/null; rsh "mkdir -p $RDIR/models $RDIR/results; chmod +x $RDIR/executor_runner $RDIR/run_models.sh"
       for f in "${models[@]}"; do adb -s "$SERIAL" push "$f" "$RDIR/models/" > /dev/null; done;;
esac
{ echo "# $(date -u +%FT%TZ) sha256 on $DEV"; rsh "cd $RDIR && sha256sum executor_runner libexecutorch_vulkan_shared_runtime.so run_models.sh models/*"; } >> "$OUT/$DEV/deployed.txt"
echo "deployed to $DEV:$RDIR"
