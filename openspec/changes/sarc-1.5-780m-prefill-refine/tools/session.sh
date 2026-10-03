#!/bin/bash
# session.sh <session> [e2e5 args]: wait for the GPU to cool (<= 48 C or 5 min), then run e2e5.sh on stage/<session>.
A=$HOME/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-03; S=$1; shift
t0=$SECONDS; while (( $(cat /sys/class/hwmon/hwmon2/temp1_input) > 48000 && SECONDS - t0 < 300 )); do sleep 5; done
$A/tools/e2e5.sh --stage $A/stage/$S --out raw --lock 00000000-c400-0000-0000-000000000000 "$@"
