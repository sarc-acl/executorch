#!/bin/bash
# collect.sh: copy the small evidence files of this study from .artifacts into the topic tree
# (openspec/changes/sarc-1.5-b580-prefill-refine/). Raw logs, ETDumps, clock samples and binaries stay in
# /mnt/linux-share/hmz-campaigns/b580/.artifacts/ (NAS).
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"
R=$C/results/b580; mkdir -p $C/tools $R/screens $R/phases

for s in $A/stage/*/; do n=$(basename $s); [[ -f $s/raw/runs.csv || -f $s/verify.out ]] || continue
  D=$R/sessions/$n; mkdir -p $D
  cp -f $s/STAGE.md $s/raw/runs.csv $s/raw/env.txt $D/ 2>/dev/null
  cp -f $s/gate.txt $s/gate.done $s/raw/done.txt $D/ 2>/dev/null
  [[ -f $s/raw/runs.csv ]] && python3 $TOOLS/summarize.py $s/raw > $D/summary.csv
  cp -f $s/raw/nexttoken.csv $D/ 2>/dev/null
  [[ -f $s/verify.out ]] && { cp -f $s/verify.out $D/; mkdir -p $D/verify; cp -f $s/verify/env.txt $s/verify/correctness.log $D/verify/ 2>/dev/null
    for q in 4w 8da4w; do grep -E '^(linear|baseline) |geomean|unexpected|confirmed|crashed' $s/verify/linear-$q.log > $D/verify/linear-$q.txt 2>/dev/null; done; }
  [[ -d $s/trace/report/evidence/trace ]] && { mkdir -p $D/trace; cp -f $s/trace/report/evidence/trace/*.csv $D/trace/; }
  [[ -d $s/sdpa-correctness ]] && cp -rf $s/sdpa-correctness $D/
  cp -f $s/decision.txt $D/ 2>/dev/null
  [[ -f $s/decode/summary.csv ]] && { mkdir -p $D/decode; cp -f $s/decode/summary.csv $s/decode/decode.csv $D/decode/; }
  # logits probe (owner decisions of 2026-10-04): the tables, not the logit vectors
  [[ -f $s/probe/summary.csv ]] && { mkdir -p $R/probe/$n; cp -f $s/probe/summary.csv $s/probe/per_prompt.csv $s/probe/differing.md $s/probe/analysis.txt $s/probe/env.txt $s/probe/prompts.txt $R/probe/$n/; }
done
cp -f $A/build/*.src.txt $A/build/*.golden.txt $R/ 2>/dev/null
for d in $A/raw/screen*/; do n=$(basename $d); [[ -f $d/summary.csv ]] && cp -f $d/summary.csv $R/screens/$n.csv; [[ -f $d/summary.txt ]] && cp -f $d/summary.txt $R/screens/$n.txt; done
for d in $A/raw/prof-*/; do n=$(basename $d); [[ -f $d/phases.csv ]] && cp -f $d/phases.csv $R/phases/$n.csv; done
for d in $A/raw/screen*/; do n=$(basename $d); [[ -f $d/screen.csv ]] && { cp -f $d/screen.csv $R/screens/$n-runs.csv; python3 $TOOLS/screen_sdpa_summary.py $d/screen.csv > $R/screens/$n.csv; }; done
# stopped or bad sessions: the reason, the run table and the session log
for d in $A/superseded/*/; do n=$(basename $d); mkdir -p $R/superseded/$n; cp -f $d/README $d/*.out $R/superseded/$n/ 2>/dev/null; find $d -name runs.csv -exec cp -f {} $R/superseded/$n/ \; ; done
find $C -type f | wc -l; du -sh $C
# roofs: the report of the igpu-roofline run used for percent-of-roof (tools/roof.sh)
for d in $A/roofline/*/b580; do n=$(basename $(dirname $d)); mkdir -p $R/roofline/$n; cp -f $d/report/REPORT.md $d/report/summary.json $d/report/sustained-runs.csv $d/fleet-metadata.json $R/roofline/$n/ 2>/dev/null; done
