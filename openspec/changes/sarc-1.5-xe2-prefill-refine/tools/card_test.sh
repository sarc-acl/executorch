#!/bin/bash
# card_test.sh <build tag> <space> [ref=<label>:<token> ...]: what does screening on
# both B70 cards at once do to a cheap screen? (Owner decision 2026-10-05; a queue job of card 0.) One identical
# batch, the first 30 configurations to run of the build's checked.csv, is screened in the cheap mode
#   (a) on b70-0 alone, (b) on the second card alone, (c) on both cards at the same time (ct-c0, ct-c1).
# Acceptance, fixed before the first run (STATUS.md, 2026-10-05): Spearman rank correlation of the layer-weighted
# score >= 0.95 and of every shape class >= 0.90, for card-to-card (a against b) and for alone against together
# on each card (a against c0, b against c1), on at least 20 configurations; and the wall time of (c) at most
# 1.5 times the longer of (a) and (b) (otherwise the split saves too little to be worth the disturbance).
# Reported without a threshold: the time ratios and how far single configurations move around them.
# Also recorded: which card's DRM client did the work in (a) and (b) (drm-<a|b>.txt: the engine counters of the
# running test process per PCI device), i.e. that ETVK_DEVICE_INDEX = XE2_CARD selects the intended card, and
# the temperature and clock of both cards every 2 s (sensors.csv).
# On CARD_SPLIT_OK the second card's queue is started (queue1/); sweep_screen.sh then splits every cheap screen.
. "$(dirname "$(readlink -f "$0")")/host.sh"; T=${1:?build tag}; S=${2:?space}; shift 2; O=$A/sweep/cardtest
[[ -e $O/verdict ]] && { echo "card test already done: $(cat $O/verdict)"; exit 0; }
mkdir -p $O
python3 - $A/sweep/$T/checked.csv $O/batch.csv <<'P'
import csv, sys
r = list(csv.DictReader(open(sys.argv[1]))); v = [x for x in r if x["run"] == "1"][:30]
w = csv.DictWriter(open(sys.argv[2], "w", newline=""), r[0].keys()); w.writeheader(); w.writerows(v)
P
D0=/sys/bus/pci/devices/0000:01:00.0; D1=/sys/bus/pci/devices/0000:02:00.0
( echo "utc,phase,temp0_c,mhz0,temp1_c,mhz1"; while :; do
    echo "$(date -u +%T),$(cat $O/phase 2>/dev/null),$(( $(cat $D0/hwmon/hwmon*/temp2_input) / 1000 )),$(cat $D0/tile0/gt0/freq0/act_freq),$(( $(cat $D1/hwmon/hwmon*/temp2_input) / 1000 )),$(cat $D1/tile0/gt0/freq0/act_freq)"
    sleep 2; done ) >> $O/sensors.csv & SENS=$!
drm() { # drm <phase>: PCI device and engine of every DRM client of a running test process with engine time, during that phase
  while [[ $(cat $O/phase) == $1 ]]; do for p in $(pgrep -x test_llama_micr); do
      awk '/^drm-pdev:/ {d = $2} /^drm-cycles-/ && $2 + 0 > 0 {print d, $1}' /proc/$p/fdinfo/* 2>/dev/null
    done; sleep 0.3; done | awk '!s[$0]++' > $O/drm-$1.txt; }
run() { XE2_CARD=$1 python3 $TOOLS/sweep_run.py ct-$2 $T $S cheap 1 $O/batch.csv "${@:3}" > $O/ct-$2.out 2>&1; }
rc=0
echo a > $O/phase; drm a & t=$SECONDS; run 0 a "$@" || rc=$?; TA=$((SECONDS - t))
echo b > $O/phase; drm b & t=$SECONDS; run 1 b "$@" || rc=$?; TB=$((SECONDS - t))
echo c > $O/phase; sleep 1; t=$SECONDS; run 0 c0 "$@" & p0=$!; run 1 c1 "$@" & p1=$!; wait $p0 || rc=$?; wait $p1 || rc=$?; TC=$((SECONDS - t))
kill $SENS; echo "a=$TA b=$TB c=$TC rc=$rc" > $O/times.txt
[[ $rc == 0 ]] || { echo "CARD_TEST_INCOMPLETE rc=$rc"; exit $rc; }
python3 $TOOLS/card_test.py $O $TA $TB $TC || exit 1
echo "DRM clients with engine time: (a) $(sort -u $O/drm-a.txt | tr '\n' ';') (b) $(sort -u $O/drm-b.txt | tr '\n' ';')"
if grep -qx CARD_SPLIT_OK $O/verdict; then
  mkdir -p $A/queue1; XE2_CARD=1 XE2_TOP= nohup setsid bash $TOOLS/sweep_queue.sh > $A/queue1/queue.out 2>&1 8>&- 7>&- &
  echo "second card's queue started"
fi
