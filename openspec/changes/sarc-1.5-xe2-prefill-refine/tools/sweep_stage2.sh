#!/bin/bash
# sweep_stage2.sh <space> [ref=<label>:<token> ...]: stage 2 of the sampled search for one space (a queue job),
# after sweep_stage1.sh:
#   1. correctness (sweep_run.py corr) of every configuration near the top of a list, repeated until the lists
#      are stable (at most 4 rounds), with the analysis (sweep_analyze.py) in between;
#   2. refinement: every unmeasured legal one-parameter neighbour of the seeds (best.csv: the 20 best by score
#      and the 5 best of each shape class), at most 600 after the compile check, in the sweep build sw2-<space>;
#      cheap screen, then step 1 again over both builds;
#   3. confirmation: the 10 best per shape class (top10.csv) rebuilt together as sw3-<space> and measured with
#      the full measurement, twice, interleaved with the reference arms.
# Analysis in sweep/an-<space>/; results in raw/sw{1,2,3}-<space>/results.csv.
. "$(dirname "$(readlink -f "$0")")/host.sh"; S=$1; shift; C=$A/sweep/cfg; AN=$A/sweep/an-$S; PY=$XE2_PYTHON
analyze() { local w=$A/sweep/sw1-$S/checked.csv r=$A/raw/sw1-$S/results.csv
  [[ -f $A/raw/sw2-$S/results.csv ]] && { w+=,$A/sweep/sw2-$S/checked.csv; r+=,$A/raw/sw2-$S/results.csv; }
  $PY $TOOLS/sweep_analyze.py $S $AN $w $r; }
correctness() { local i b
  for i in 1 2 3 4; do analyze || return 1; [[ -s $AN/tocheck.txt ]] || return 0
    for b in sw1 sw2; do [[ -f $A/sweep/$b-$S/checked.csv ]] || continue
      python3 $TOOLS/sweep.py subset $AN/tocheck.txt $C/$S-corr-$b-$i.csv $A/sweep/$b-$S/checked.csv
      [[ $(wc -l < $C/$S-corr-$b-$i.csv) -gt 1 ]] && { python3 $TOOLS/sweep_run.py $b-$S $b-$S $S corr 1 $C/$S-corr-$b-$i.csv || return $?; }
    done; done; analyze; }
correctness || exit $?
if [[ ! -f $A/build/sw2-$S.src.txt ]]; then
  cp $AN/best.csv $C/$S-seeds.csv
  python3 $TOOLS/sweep.py neighbours $S $A/sweep/sw1-$S/checked.csv $C/$S-seeds.csv 50000 $C/$S-neighbours.csv || exit 1
  bash $TOOLS/build-sweep.sh sw2-$S 600 $C/$S-neighbours.csv || exit 1
fi
python3 $TOOLS/sweep_run.py sw2-$S sw2-$S $S cheap 1 $A/sweep/sw2-$S/checked.csv "$@" || exit $?
correctness || exit $?
if [[ ! -f $A/build/sw3-$S.src.txt ]]; then
  python3 $TOOLS/sweep.py subset $AN/top10.csv $C/$S-finalists.csv $A/sweep/sw1-$S/checked.csv $A/sweep/sw2-$S/checked.csv || exit 1
  bash $TOOLS/build-sweep.sh sw3-$S 0 $C/$S-finalists.csv || exit 1
fi
python3 $TOOLS/sweep_run.py sw3-$S sw3-$S $S full 2 $A/sweep/sw3-$S/checked.csv "$@" || exit $?
