#!/bin/bash
# collect.sh: copy the small evidence files of this study from .artifacts into the topic tree
# (openspec/changes/sarc-1.5-780m-prefill-refine/). Raw logs, ETDumps, clock samples and binaries stay in
# ~/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-03/ on rocky-ryzen.
set -uo pipefail
A=$HOME/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-03; C=$HOME/hmz-sarc/executorch/openspec/changes/sarc-1.5-780m-prefill-refine
R=$C/results/780m; mkdir -p $C/tools $R/screens $R/phases
cp -f $A/tools/*.sh $A/tools/*.py $A/tools/Containerfile $C/tools/
for s in $A/stage/*/; do n=$(basename $s); [[ -f $s/raw/runs.csv ]] || continue
  D=$R/sessions/$n; mkdir -p $D
  cp -f $s/STAGE.md $s/raw/runs.csv $s/raw/env.txt $D/ 2>/dev/null
  [[ -f $s/raw/done.txt ]] && python3 $A/tools/summarize.py $s/raw > $D/summary.csv
  cp -f $s/raw/nexttoken.csv $D/ 2>/dev/null
  [[ -f $s/verify.out ]] && { cp -f $s/verify.out $D/; mkdir -p $D/verify; cp -f $s/verify/env.txt $s/verify/correctness.log $D/verify/ 2>/dev/null
    for q in 4w 8da4w; do grep -E '^(linear|baseline) |geomean|unexpected|confirmed|crashed' $s/verify/linear-$q.log > $D/verify/linear-$q.txt 2>/dev/null; done; }
  [[ -d $s/trace/report/evidence/trace ]] && { mkdir -p $D/trace; cp -f $s/trace/report/evidence/trace/*.csv $D/trace/; }
  [[ -d $s/sdpa-correctness ]] && cp -rf $s/sdpa-correctness $D/
done
for d in $A/raw/screen*/; do n=$(basename $d); [[ -f $d/summary.csv ]] && cp -f $d/summary.csv $R/screens/$n.csv; [[ -f $d/summary.txt ]] && cp -f $d/summary.txt $R/screens/$n.txt; done
for d in $A/raw/prof-*/; do n=$(basename $d); for q in 4w 8da4w; do [[ -f $d/$q/phases.csv ]] && cp -f $d/$q/phases.csv $R/phases/$n-$q.csv; done; done
find $C -type f | wc -l; du -sh $C
