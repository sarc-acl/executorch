#!/bin/bash
# sweep_refine.sh <space> <round: b | c> [ref=<label>:<token> ...]: a further refinement round of the sampled
# search (a queue job), after sweep_stage2.sh (round 1 = builds sw2-<space>, sw3-<space>). Run only when the
# previous round moved the best score or the best of a shape class by more than 2 %; at most rounds b and c.
#   1. the analysis over every build so far; the unmeasured legal one-parameter neighbours of its seeds
#      (best.csv), at most 600 after the compile check, in the sweep build sw2<round>-<space>; cheap screen;
#   2. correctness of everything near the top (as in stage 2), over every build;
#   3. confirmation: the 10 best per shape class rebuilt as sw3<round>-<space>, full measurement twice.
. "$(dirname "$(readlink -f "$0")")/host.sh"; S=$1; R=$2; shift 2; C=$A/sweep/cfg; AN=$A/sweep/an-$S; PY=$XE2_PYTHON
[[ $R == b || $R == c ]] || { echo "round must be b or c" >&2; exit 2; }
FIRST=60000; [[ $R == c ]] && FIRST=70000
builds() { local b; for b in sw1 sw2 sw2b sw2c; do [[ -f $A/raw/$b-$S/results.csv ]] && echo $b; done; }
analyze() { local w= r= b; for b in $(builds); do w+=${w:+,}$A/sweep/$b-$S/checked.csv; r+=${r:+,}$A/raw/$b-$S/results.csv; done
  $PY $TOOLS/sweep_analyze.py $S $AN $w $r; }
correctness() { local i b
  for i in 1 2 3 4; do analyze || return 1; [[ -s $AN/tocheck.txt ]] || return 0
    for b in $(builds); do
      python3 $TOOLS/sweep.py subset $AN/tocheck.txt $C/$S-corr$R-$b-$i.csv $A/sweep/$b-$S/checked.csv
      [[ $(wc -l < $C/$S-corr$R-$b-$i.csv) -gt 1 ]] && { python3 $TOOLS/sweep_run.py $b-$S $b-$S $S corr 1 $C/$S-corr$R-$b-$i.csv || return $?; }
    done; done; analyze; }
if [[ ! -f $A/build/sw2$R-$S.src.txt ]]; then
  analyze || exit 1; cp $AN/best.csv $C/$S-seeds$R.csv
  { head -1 $A/sweep/sw1-$S/checked.csv; for b in $(builds); do tail -n +2 $A/sweep/$b-$S/checked.csv; done; } > $C/$S-measured$R.csv
  python3 $TOOLS/sweep.py neighbours $S $C/$S-measured$R.csv $C/$S-seeds$R.csv $FIRST $C/$S-neighbours$R.csv || exit 1
  bash $TOOLS/build-sweep.sh sw2$R-$S 600 $C/$S-neighbours$R.csv || exit 1
fi
python3 $TOOLS/sweep_run.py sw2$R-$S sw2$R-$S $S cheap 1 $A/sweep/sw2$R-$S/checked.csv "$@" || exit $?
correctness || exit $?
if [[ ! -f $A/build/sw3$R-$S.src.txt ]]; then
  W=(); for b in $(builds); do W+=($A/sweep/$b-$S/checked.csv); done
  python3 $TOOLS/sweep.py subset $AN/top10.csv $C/$S-finalists$R.csv "${W[@]}" || exit 1
  bash $TOOLS/build-sweep.sh sw3$R-$S 0 $C/$S-finalists$R.csv || exit 1
fi
python3 $TOOLS/sweep_run.py sw3$R-$S sw3$R-$S $S full 2 $A/sweep/sw3$R-$S/checked.csv "$@" || exit $?
