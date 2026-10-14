#!/bin/bash
# sweep_collect.sh <space>...: copy the evidence of the sampled search from the artifact directory into
# results/xe2/sweep/<space>/ (analysis tables, and per sweep build its provenance, configuration list, result
# rows, runner log and, where the second card screened half of it, that card's own rows and the scale factors),
# and the card test into results/xe2/sweep/cardtest/. CPU only; run it whenever a stage has ended.
. "$(dirname "$(readlink -f "$0")")/host.sh"
# importance-sample.csv / interactions-sample.csv are the same analysis over the uniform sample (or the full
# enumeration) alone, build sw1: the refinement neighbours cluster around the best and would bias the shares.
for S in "$@"; do D=$C/results/xe2/sweep/$S; mkdir -p $D; cp $A/sweep/an-$S/* $D/ 2>/dev/null
  if [[ -f $A/raw/sw1-$S/results.csv ]]; then T=$(mktemp -d)
    $XE2_PYTHON $TOOLS/sweep_analyze.py $S $T $A/sweep/sw1-$S/checked.csv $A/raw/sw1-$S/results.csv > $D/analysis-sample.txt 2>&1
    cp $T/importance.csv $D/importance-sample.csv; cp $T/interactions.csv $D/interactions-sample.csv; rm -rf $T; fi
  for p in $A/build/sw*-$S.src.txt; do [[ -e $p ]] || continue; b=$(basename $p .src.txt); b=${b%-$S}
    cp $p $D/$b-build.txt; cp $A/sweep/$b-$S/checked.csv $D/$b-checked.csv
    [[ -f $A/raw/$b-$S/results.csv ]] && { cp $A/raw/$b-$S/results.csv $D/$b-results.csv; cp $A/raw/$b-$S/env.txt $D/$b-env.txt; }
    [[ -f $A/raw/$b-$S-c1/results.csv ]] && { cp $A/raw/$b-$S-c1/results.csv $D/$b-results-card1.csv; cp $A/raw/$b-$S-c1/env.txt $D/$b-env-card1.txt
                                               cp $A/raw/$b-$S/card1-scale.csv $D/$b-card1-scale.csv 2>/dev/null; }
  done; done
if [[ -e $A/sweep/cardtest/verdict ]]; then D=$C/results/xe2/sweep/cardtest; mkdir -p $D
  cp $A/sweep/cardtest/{batch.csv,cardtest.csv,report.txt,verdict,times.txt,drm-a.txt,drm-b.txt,sensors.csv} $D/
  for n in a b c0 c1; do cp $A/raw/ct-$n/results.csv $D/ct-$n-results.csv; done; fi
cp $A/queue/status $C/results/xe2/sweep/queue-status.txt; [[ -f $A/queue1/status ]] && cp $A/queue1/status $C/results/xe2/sweep/queue1-status.txt; true
