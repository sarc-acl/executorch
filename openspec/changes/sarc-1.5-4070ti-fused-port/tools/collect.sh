#!/bin/bash
# collect.sh: copy the small evidence files of this study from .artifacts into the topic tree
# (openspec/changes/sarc-1.5-4070ti-fused-port/). Raw logs, ETDumps, clock samples and binaries stay in the
# artifact directory on the GPU host. The tools live in the tree and are run from there.
set -uo pipefail
source "$(dirname "$0")/common.sh"; C=$CHANGE
R=$C/results/4070ti; mkdir -p $R/screens $R/phases
for s in $A/stage/*/; do n=$(basename $s); [[ -f $s/raw/runs.csv || -f $s/verify.out ]] || continue
  D=$R/sessions/$n; mkdir -p $D
  cp -f $s/STAGE.md $s/raw/runs.csv $s/raw/env.txt $D/ 2>/dev/null
  cp -f $s/gate.done* $s/*-check.txt $s/verify-runs.jsonl $D/ 2>/dev/null
  cp -f $s/control.done $s/control.diff $s/verify-warm.csv $s/raw/warm.csv $s/raw/loads.csv $D/ 2>/dev/null
  [[ -f $s/trace/raw/4070ti/trace2/warm.csv ]] && { mkdir -p $D/trace; cp -f $s/trace/raw/4070ti/trace2/warm.csv $D/trace/warm.csv; cp -f $s/trace/raw/4070ti/trace2/wall.csv $D/trace/wall.csv; }
  [[ -f $s/raw/done.txt ]] && python3 "$(dirname "$0")/summarize.py" $s/raw $(sed -n 's/.* reps=\([0-9]*\) .*/\1/p' $s/raw/env.txt | head -1) > $D/summary.csv
  cp -f $s/raw/nexttoken.csv $D/ 2>/dev/null
  [[ -f $s/verify.out ]] && { cp -f $s/verify.out $D/; mkdir -p $D/verify; cp -f $s/verify/env.txt $s/verify/correctness.log $D/verify/ 2>/dev/null
    for q in 4w 8da4w; do grep -E '^(linear|baseline) |geomean|unexpected|confirmed|crashed' $s/verify/linear-$q.log > $D/verify/linear-$q.txt 2>/dev/null; done; }
  [[ -d $s/trace/report/evidence/trace ]] && { mkdir -p $D/trace; cp -f $s/trace/report/evidence/trace/*.csv $D/trace/; }
  [[ -d $s/sdpa-correctness ]] && cp -rf $s/sdpa-correctness $D/
done
for d in $A/raw/screen*/; do n=$(basename $d); [[ -f $d/summary.csv ]] && cp -f $d/summary.csv $R/screens/$n.csv; [[ -f $d/summary.txt ]] && cp -f $d/summary.txt $R/screens/$n.txt; done
for d in $A/raw/first-use-*/; do [[ -f $d/rows.csv ]] && { mkdir -p $R/exit-probe; cp -f $d/rows.csv $R/exit-probe/$(basename $d).csv; }; done
for d in $A/raw/prof-*/; do n=$(basename $d); for q in 4w 8da4w; do [[ -f $d/$q/phases.csv ]] && cp -f $d/$q/phases.csv $R/phases/$n-$q.csv; done; done
find $C -type f | wc -l; du -sh $C
