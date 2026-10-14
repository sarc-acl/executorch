#!/bin/bash
# collect.sh: workstation side, after pull.sh. Copies the small evidence files of this study from the device
# mirror (<artifacts>/device/) into the topic tree (results/orin/). Raw logs, ETDumps, clock samples and binaries
# stay in the artifact directory and on the device.
set -uo pipefail
source "$(dirname "$0")/common.sh"; D=$A/device; R=$CHANGE/results/orin; mkdir -p $R/screens $R/sessions
for s in $D/stage/*/; do n=$(basename $s); [[ -f $s/raw/runs.csv || -f $s/verify.out ]] || continue
  O=$R/sessions/$n; mkdir -p $O
  cp -f $s/STAGE.md $s/raw/runs.csv $s/raw/env.txt $O/ 2>/dev/null
  cp -f $s/gate.done* $s/*-check*.txt $s/verify-runs.jsonl $s/gate.env $O/ 2>/dev/null
  [[ -f $s/raw/done.txt ]] && python3 "$TOOLS/summarize.py" $s/raw > $O/summary.csv
  cp -f $s/raw/nexttoken.csv $O/ 2>/dev/null
  [[ -f $s/verify.out ]] && { cp -f $s/verify.out $O/; mkdir -p $O/verify; cp -f $s/verify/env.txt $s/verify/correctness.log $O/verify/ 2>/dev/null
    for q in 4w 8da4w; do grep -E '^(linear|baseline) |geomean|unexpected|confirmed|crashed' $s/verify/linear-$q.log > $O/verify/linear-$q.txt 2>/dev/null; done
    grep -h '^\[production-diff\]\|total mismatched' $s/verify/pdiff-*.log > $O/verify/pdiff.txt 2>/dev/null; }
  [[ -d $s/trace/report/evidence/trace ]] && { mkdir -p $O/trace; cp -f $s/trace/report/evidence/trace/*.csv $O/trace/; }
  if [[ -d $s/sdpa-correctness ]]; then mkdir -p $O/sdpa-correctness; cp -f $s/sdpa-correctness/*.txt $O/sdpa-correctness/ 2>/dev/null
    for f in $s/sdpa-correctness/*.log; do grep -h '^\[sdpa-correctness\]\|^\[sdpa-error\]\|^\[sdpa-kernels\]\|^\[sarc_dev\]' $f | cut -c1-400 > $O/sdpa-correctness/$(basename $f .log).txt; done; fi
done
for d in $D/raw/*screen*/; do n=$(basename $d); [[ -f $d/rows.csv ]] && { cp -f $d/rows.csv $R/screens/$n-rows.csv; python3 "$TOOLS/sdpa_screen_summary.py" $d > $R/screens/$n.csv; }; done
for d in $D/raw/c*-pre/; do [[ -d $d ]] || continue; n=$(basename $d); mkdir -p $R/$n; cp -f $d/summary.txt $R/$n/ 2>/dev/null
  for f in $d/*.log; do grep -h '^\[sdpa-correctness\]\|^\[sdpa-error\]\|^\[sdpa-kernels\]\|^\[sarc_dev\]' $f | cut -c1-400 > $R/$n/$(basename $f .log).txt; done; done
for d in $D/probe/*-fused*/; do [[ -d $d ]] || continue; n=$(basename $d); mkdir -p $R/probe/$n/position; cp -f $d/*.txt $d/*.csv $d/*.json $d/cand.env $R/probe/$n/ 2>/dev/null; cp -f $d/position/* $R/probe/$n/position/ 2>/dev/null; done
for d in $D/raw/prof*/; do n=$(basename $d); for q in 4w 8da4w; do [[ -f $d/$q/phases.csv ]] && cp -f $d/$q/phases.csv $R/phases/$n-$q.csv; done; done
for d in $D/raw/sdpa-error*/; do [[ -d $d ]] || continue; n=$(basename $d); mkdir -p $R/$n; cp -f $d/*.txt $R/$n/ 2>/dev/null; done
find $CHANGE -type f | wc -l; du -sh $CHANGE
