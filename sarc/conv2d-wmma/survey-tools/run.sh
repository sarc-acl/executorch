#!/bin/bash
# run.sh DEVICE NAME...: runs run_models.sh in the device directory (each run under the device's gpu-lab lock,
# Vulkan device 0) and records the command in OUT/DEVICE/run-commands.txt. Then fetch.sh copies results back.
set -u
. "$(dirname "$0")/devices.sh"; device_info "$1"; DEV=$1; shift
CMD="cd $RDIR && sh ./run_models.sh $LOCKFILE $*"
echo "# $(date -u +%FT%TZ) $DEV ($KIND${HOST:+ $HOST}${SERIAL:+ $SERIAL})" >> "$OUT/$DEV/run-commands.txt"
echo "$CMD" >> "$OUT/$DEV/run-commands.txt"
rsh "$CMD"
"$(dirname "$0")/fetch.sh" "$DEV"
