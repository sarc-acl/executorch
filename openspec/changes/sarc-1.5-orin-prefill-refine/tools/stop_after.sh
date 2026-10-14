#!/bin/bash
# stop_after.sh <job> <file> <reason>: device side. Waits until <file> exists (a step of <job> has completed), then
# ends <job> by its recorded session id before its next step does any work, and records why. Used to re-order a
# running chain at a step boundary without interrupting a measurement.
source "$(dirname "$0")/common.sh"; J=$1; F=$2; shift 2; [[ $F == /* ]] || F=$A/$F   # relative to ~/hmz-sarc-orin
while [[ ! -e $F ]]; do grep -q "^DONE\|^KILLED" $A/jobs/$J.status && { echo "job $J ended by itself"; exit 0; }; sleep 5; done
sleep 2; p=$(cat $A/jobs/$J.pid); pkill -TERM -s $p; sleep 1
echo "KILLED $(date -u +%FT%TZ) at a step boundary ($(basename $F) written): $*" >> $A/jobs/$J.status; tail -1 $A/jobs/$J.status
