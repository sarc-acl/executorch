#!/bin/bash
# roof.sh <name>: device side. Fresh roofs with igpu-roofline's `fast` plan (roof_fast.py) into raw/roof-<name>/,
# under the gpu-lab lock. The roofline tree is a private copy of ~/.cache/igpu-roofline/fleet-quick-20260925
# (without its results). Clock, power mode and temperature are recorded before and after; nothing is changed.
source "$(dirname "$0")/common.sh"; [[ $SIDE == device ]] || { echo "device only" >&2; exit 2; }
SRC=$HOME/.cache/igpu-roofline/fleet-quick-20260925; RT=$A/roofline; O=$A/raw/roof-$1
[[ -e $O ]] && { echo "$O exists" >&2; exit 2; }
if [[ ! -d $RT ]]; then mkdir -p $RT; rsync -a --exclude /results --exclude /controller.log --exclude /stage $SRC/ $RT/; mkdir -p $RT/stage; fi
mkdir -p $O
snap() { echo "$(date -u +%FT%TZ) clk=$(gclk)MHz min=$(cat $GPUDEV/min_freq) max=$(cat $GPUDEV/max_freq) gov=$(cat $GPUDEV/governor) temp=$(gtemp_m)mC $(nvpmodel -q 2>/dev/null | tr '\n' ' ') $(mem_line)"; }
{ echo "source $SRC"; cat $SRC/campaign-config.json | grep -m1 source_commit; sha256sum $RT/build/host/roofline $RT/build/host/inspect; snap; } > $O/env.txt
( while :; do snap; sleep 10; done > $O/clock-observations.txt ) & MON=$!
cool_start 300
"$TOOLS/gl.sh" python3 -u "$TOOLS/roof_fast.py" $RT $O > $O/run.log 2>&1; rc=$?
kill $MON 2>/dev/null; snap >> $O/env.txt; echo "ROOF_DONE rc=$rc"; exit $rc
