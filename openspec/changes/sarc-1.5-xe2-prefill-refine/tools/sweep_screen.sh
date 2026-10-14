#!/bin/bash
# sweep_screen.sh <build tag> <space> <checked.csv> [ref=<label>:<token> ...]: the cheap screen of a sweep build
# (called by the stage scripts, on card 0's queue). Results: raw/<tag>/results.csv.
# With the second card accepted for screening (sweep/cardtest/verdict = CARD_SPLIT_OK, written by card_test.sh)
# and its queue running, the list is split by position: the odd rows go to the second card as a job of its
# queue (raw/<tag>-c1/), the even rows run here at the same time; when the other half has ended, its rows are
# scaled by the base arm of both cards and appended (sweep_merge.py). Then the whole list is run here once
# more, which measures only what is still missing (everything, if the second card was not used or stopped).
. "$(dirname "$(readlink -f "$0")")/host.sh"; T=$1; S=$2; W=$3; shift 3; Q1=$A/queue1; J=$Q1/pending/$T-c1.sh
alive() { ! ( exec 8>>$Q1/lock; flock -n 8 ) 2>/dev/null; }
if grep -qx CARD_SPLIT_OK $A/sweep/cardtest/verdict 2>/dev/null && [[ -d $Q1/pending ]] && alive; then
  if [[ ! -e $J && ! -e $Q1/done/$T-c1.sh ]]; then
    { printf 'python3 %s/sweep_run.py %s-c1 %s %s cheap 1 %s part=1/2' $TOOLS $T $T $S $W; printf ' %q' "$@"; echo; } > $J.tmp; mv $J.tmp $J
  fi
  python3 $TOOLS/sweep_run.py $T $T $S cheap 1 $W part=0/2 "$@" || exit $?
  while [[ -e $J ]] && alive; do sleep 20; done
  [[ -e $J ]] && { mv $J $Q1/withdrawn-$T-c1.sh; echo "the second card's queue is not running: its half is measured here"; }
  if [[ -f $A/raw/$T-c1/results.csv ]]; then
    python3 $TOOLS/sweep_merge.py $A/raw/$T/results.csv $A/raw/$T-c1/results.csv $A/raw/$T/card1-scale.csv || exit 1
  fi
fi
python3 $TOOLS/sweep_run.py $T $T $S cheap 1 $W "$@"
