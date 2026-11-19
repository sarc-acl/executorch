#!/bin/bash
# fetch.sh DEVICE: copies the device directory's results/ into OUT/DEVICE/results/.
set -eu
. "$(dirname "$0")/devices.sh"; device_info "$1"; DEV=$1
mkdir -p "$OUT/$DEV/results"
case $KIND in
  local) ;;
  ssh) scp -q -r "$HOST:$RDIR/results/." "$OUT/$DEV/results/";;
  adb) adb -s "$SERIAL" pull "$RDIR/results/." "$OUT/$DEV/results/" > /dev/null;;
esac
echo "fetched $DEV: $(ls "$OUT/$DEV/results" | wc -l) files"
