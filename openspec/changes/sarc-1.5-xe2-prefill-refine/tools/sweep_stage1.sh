#!/bin/bash
# sweep_stage1.sh <space> <draws> <keep> [ref=<label>:<token> ...]: stage 1 of the sampled search for one space
# (a queue job): the seeded sample, the sweep build sw1-<space> with its compile check, then on the card
#   1. the validation set (the first 60 configurations to run): cheap, then the full measurement twice;
#   2. the cheap screen of every configuration to run.
# Seed 20261005 for every space. Results: raw/sw1-<space>/results.csv.
. "$(dirname "$(readlink -f "$0")")/host.sh"; S=$1; D=$2; K=$3; shift 3; C=$A/sweep/cfg; mkdir -p $C
[[ -f $C/$S-sample.csv ]] || python3 $TOOLS/sweep.py sample $S 20261005 $D $C/$S-sample.csv || exit 1
[[ -f $A/build/sw1-$S.src.txt ]] || bash $TOOLS/build-sweep.sh sw1-$S $K $C/$S-sample.csv || exit 1
grep -qx BUILD_SWEEP_OK $A/build/sw1-$S.src.txt || { echo "build sw1-$S failed"; exit 1; }
W=$A/sweep/sw1-$S/checked.csv
python3 - $W $C/$S-val.csv <<'P'
import csv, sys
r = list(csv.DictReader(open(sys.argv[1]))); v = [x for x in r if x["run"] == "1"][:60]
w = csv.DictWriter(open(sys.argv[2], "w", newline=""), r[0].keys()); w.writeheader(); w.writerows(v)
P
python3 $TOOLS/sweep_run.py sw1-$S sw1-$S $S cheap 1 $C/$S-val.csv "$@" || exit $?
python3 $TOOLS/sweep_run.py sw1-$S sw1-$S $S full 2 $C/$S-val.csv "$@" || exit $?
bash $TOOLS/sweep_screen.sh sw1-$S $S $W "$@" || exit $?
