#!/bin/bash
# dstat.sh [job]: workstation side. Status of the device jobs (all, or the tail of one job's output), what is
# running there now, temperature, clock, memory, and the abort / device-lost markers.
source "$(dirname "$0")/common.sh"
dssh "cd $DEVROOT; for f in jobs/*.status; do echo \"\$(basename \$f .status): \$(tail -1 \$f | cut -c1-150)\"; done 2>/dev/null | tail -${N:-8}
[[ -n '${1:-}' ]] && tail -${L:-15} jobs/$1.out
pgrep -u \$USER -a 'llama_main|test_llama_micr|logits_dump' | cut -c1-160
echo \"gpu \$(( \$(cat /sys/class/thermal/thermal_zone1/temp) / 1000 )) C, \$(( \$(cat /sys/class/devfreq/17000000.gpu/cur_freq) / 1000000 )) MHz, \$(free -m | awk '/Mem:/ {print \"avail \" \$7 \" MB\"} /Swap:/ {print \"swap used \" \$3 \" MB\"}' | tr '\n' ' ')\$(date -u +%T)\"
cat ABORTED GPU_GONE 2>/dev/null | tail -3"
