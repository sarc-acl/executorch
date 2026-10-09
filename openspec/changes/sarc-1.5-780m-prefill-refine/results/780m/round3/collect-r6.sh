#!/bin/bash
# collect-r6.sh: copy the small evidence files of the monitored timed sessions (chain29.sh stopped, chain30.sh;
# owner decision 2026-10-09 02:50 UTC) from <artifacts 10-08> into results/780m/. Logs, .clk and .mon stay there.
set -uo pipefail
A=$HOME/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-08; C=$HOME/hmz-sarc/executorch/openspec/changes/sarc-1.5-780m-prefill-refine
R=$C/results/780m; cp -f $A/logs/chain29.status $A/logs/chain30.status $R/round3/
for n in r3a-fused3sb-head4 r3b-final-dev15-head4; do s=$A/stage/$n; D=$R/sessions/$n-r6; mkdir -p $D
  cp -f $s/raw-r6b/runs.csv $s/raw-r6b/env.txt $s/raw-r6b/summary.csv $D/; cp -f $s/prestart-r6b.txt $D/prestart.txt
  sed -i "s|$A/||g" $D/env.txt
  python3 $C/tools/r6_analyze.py $s/raw-r6b > $D/analysis.txt
done
D=$R/sessions/r3a-fused3sb-head4-r6/superseded/r6-actor-shell-matched-guard; mkdir -p $D
cp -f $A/stage/r3a-fused3sb-head4/superseded/r6-actor-shell-matched-guard/raw-r6/runs.csv $D/
D=$R/round3/e2e5-smoke-r6; mkdir -p $D; cp -f $A/stage/r3a-fused3sb-head4/smoke-r6/runs.csv $D/
