#!/bin/bash
# collect.sh: copy the evidence of chain.sh from the artifact directory into data/ and recompute the tables.
set -uo pipefail
ME=$(dirname "$(readlink -f "$0")"); D=$ME/../data; CT=$ME/../../sarc-1.5-b580-fused-port/tools
A=/mnt/linux-share/hmz-campaigns/f1-b580/.artifacts; OLD=/mnt/linux-share/hmz-campaigns/b580-fused/.artifacts
mkdir -p $D/validation $D/kernel $D/build
cp $A/logs/chain.status $A/logs/idle_wait.log $A/raw/spirv-identity.txt $D/
cp $A/build/f1.src.txt $A/build/topic7.src.txt $A/build/f1.golden.txt $D/build/
cp $A/raw/validation/* $D/validation/
for b in topic7 f1; do echo "== $b"; cat $D/validation/$b-linear.log $D/validation/$b-sdpa.log | grep -o 'VUID-[A-Za-z0-9_-]*' | sort | uniq -c; done > $D/validation/summary.txt
cp $A/raw/kernel/rc.csv $A/raw/kernel/*.json $D/kernel/
python3 $ME/kernel_table.py $A/raw/kernel > $D/kernel/kernel.csv 2> $D/kernel/summary.txt
for s in sA-fused1 sA2-fused1 sB-noenv; do S=$A/stage/$s; mkdir -p $D/$s
  cp $S/STAGE.md $S/raw/runs.csv $S/raw/env.txt $S/raw/done.txt $S/raw/nexttoken.csv $S/raw/summary.csv $D/$s/
  for f in gate.txt gate-control.txt gate.env verify.out sdpa.out; do [[ -f $S/$f ]] && cp $S/$f $D/$s/; done
  [[ -d $S/sdpa-correctness ]] && cp $S/sdpa-correctness/rc.csv $D/$s/sdpa-rc.csv
  python3 $ME/session_table.py $D/$s/runs.csv > $D/$s/table.md
  [[ -f $S/probe/identity.txt ]] && cp $S/probe/identity.txt $D/$s/logits-identity.txt && cp $S/probe/analysis.txt $D/$s/logits-analysis.txt
done
python3 $ME/sdpa_records.py $A/stage/sA-fused1 cand > $D/sA-fused1/sdpa-records.txt
python3 $ME/sdpa_records.py $A/stage/sB-noenv table > $D/sB-noenv/sdpa-records.txt
python3 $CT/verify_diff.py $OLD/stage/s6-final/verify.out $A/stage/sA-fused1/verify.out > $D/sA-fused1/verify-diff.txt
python3 $CT/verify_diff.py $OLD/stage/s0-topic7-noenv/verify.out $A/stage/sB-noenv/verify.out > $D/sB-noenv/verify-diff.txt
du -sh $D; tail -1 $D/*/verify-diff.txt $D/*/sdpa-records.txt; cat $D/kernel/summary.txt
