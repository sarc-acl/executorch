#!/bin/bash
# sweep_rounds.sh <space> [ref=<label>:<token> ...]: the further refinement rounds of the sampled search, decided
# by rule (a queue job, after sweep_stage2.sh). Round b runs if round 1 (sw2) moved the best cheap score or the
# best of a shape class by more than 2 % over the sample (sw1); round c if round b moved them by more than 2 %
# over round 1, or if the full confirmation of round b (sw3b) has a (model, shape) whose best configuration is
# more than 2 % faster than the best of round 1's confirmation (sw3); there is no round d. The cheap comparison
# is sweep_analyze.py over the builds before and after the round (rounds.txt records every number used). Then the confirmation tables of every round (sweep_confirm.py against the
# incumbent arm: the first ref, or base) are written into sweep/an-<space>/.
. "$(dirname "$(readlink -f "$0")")/host.sh"; S=$1; shift; AN=$A/sweep/an-$S; PY=$XE2_PYTHON; T=$(mktemp -d)
best() { local w= r= b; for b in "$@"; do [[ -f $A/raw/$b-$S/results.csv ]] || continue; w+=${w:+,}$A/sweep/$b-$S/checked.csv; r+=${r:+,}$A/raw/$b-$S/results.csv; done
  $PY $TOOLS/sweep_analyze.py $S $T $w $r | sed -n 's/^best score_x \([0-9.]*\).*/score \1/p; s/^best \([a-z0-9_]*\): \([0-9.]*\)x.*/\1 \2/p'; }
moved() { join <(best "${@:2:$1}" | sort) <(best "${@:2}" | sort) | tee -a $AN/rounds.txt | awk '$3 > 1.02 * $2 {f = 1} END {exit !f}'; }
INC=base; for a in "$@"; do [[ $a == ref=* ]] && { INC=ref-${a#ref=}; INC=${INC%%:*}; break; }; done
confirm() { python3 $TOOLS/sweep_confirm.py $S $A/sweep/sw3$1-$S/checked.csv $A/raw/sw3$1-$S/results.csv $AN/confirm${1:+-$1}.csv $INC > $AN/confirm${1:+-$1}.txt 2>&1; }
cmoved() { confirm ""; confirm b; python3 -c '
import csv, sys
def best(p):
    r = [x for x in csv.DictReader(open(p)) if x["arm"].isdigit()]
    return {k: max(float(x[k]) for x in r if x[k]) for k in r[0] if k.endswith("_x") and any(x[k] for x in r)}
a, b = best(sys.argv[1]), best(sys.argv[2])
for k in sorted(a):
    if k in b: print(k, a[k], b[k], "MOVED" if b[k] > 1.02 * a[k] else "")
' $AN/confirm.csv $AN/confirm-b.csv | tee -a $AN/rounds.txt | grep -q MOVED; }
for R in b c; do
  if [[ ! -f $A/raw/sw3$R-$S/results.csv ]] || ! grep -q SWEEP_DONE $A/raw/sw3$R-$S/env.txt; then
    if [[ $R == b ]]; then echo "round b? best x of sw1, of sw1+sw2 (+sw2i, the incumbents and their neighbours, where it exists):" >> $AN/rounds.txt; moved 1 sw1 sw2 sw2i || break
    else echo "round c? best x of sw1+sw2, of sw1+sw2+sw2b:" >> $AN/rounds.txt; moved 3 sw1 sw2 sw2i sw2b; m=$?
      echo "confirmation, best x per (model, shape) of sw3, of sw3b:" >> $AN/rounds.txt; cmoved || [[ $m == 0 ]] || break; fi
    echo "round $R runs" >> $AN/rounds.txt
    bash $TOOLS/sweep_refine.sh $S $R "$@" || exit $?
  fi
done
for R in "" b c; do [[ -f $A/raw/sw3$R-$S/results.csv ]] && confirm "$R"; done; rm -rf $T; cat $AN/rounds.txt $AN/confirm*.txt
