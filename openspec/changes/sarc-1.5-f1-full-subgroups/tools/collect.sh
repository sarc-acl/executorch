#!/bin/bash
# collect.sh: workstation side. Mirrors the device results (the campaign's pull.sh, into <artifacts>/device/),
# copies the evidence of f1-chain.sh into data/ and recomputes the tables. The old build's verify.out of each
# configuration is read from the campaign's device mirror (s5-c1: topic4 with orin-fused1; s7n-noenv: topic4
# with no environment).
set -uo pipefail
ME=$(dirname "$(readlink -f "$0")"); D=$ME/../data
A=/mnt/linux-share/hmz-campaigns/f1-orin/.artifacts; OLD=/mnt/linux-share/hmz-campaigns/jetson-fused/.artifacts/device
$A/tools/pull.sh > /dev/null; M=$A/device
mkdir -p $D/flagproof $D/kernel $D/build $D/logits
cp $M/chain.status $D/; cp $M/jobs/chain*.out $M/jobs/chain*.status $D/ 2>/dev/null
cp $M/raw/flagproof/*.log $D/flagproof/ 2>/dev/null
for f in $D/flagproof/*.log; do echo "== $(basename $f)"; grep -h '^\[f1-flagprint\]' $f | sort | uniq -c; done > $D/flagproof/summary.txt
strip() { sed -E 's/prefill_tok_s=[0-9.]+/prefill_tok_s=X/; s/decode_tok_s=[0-9.]+/decode_tok_s=X/' "$1"; }
vdiff() { # vdiff <old verify.out> <new verify.out>: line by line with the rates removed (as noenv_verify.sh does)
  diff <(strip $1) <(strip $2); echo "old $1 ($(wc -l < $1) lines) against new $2 ($(wc -l < $2) lines), rates removed: $(diff <(strip $1) <(strip $2) | grep -c '^[<>]') differing lines"; }
for s in sA-fused1 sA2-fused1 sB-noenv sB2-noenv sBn-noenv; do S=$M/stage/$s; [[ -d $S ]] || continue; mkdir -p $D/$s
  for f in STAGE.md gate.done gate.env verify.out verify-runs.jsonl verify-check.txt verify-lines.txt verify-lines.diff sdpa-check.txt session-check.txt env-check.txt e2e5.out trace.out; do [[ -f $S/$f ]] && cp $S/$f $D/$s/; done
  for f in runs.csv env.txt done.txt nexttoken.csv summary.csv; do [[ -f $S/raw/$f ]] && cp $S/raw/$f $D/$s/; done
  [[ -f $D/$s/runs.csv ]] && python3 $ME/session_table.py $D/$s/runs.csv > $D/$s/table.md
  [[ -d $S/sdpa-correctness ]] && { cp $S/sdpa-correctness/summary.txt $D/$s/sdpa-summary.txt
    for f in $S/sdpa-correctness/cand-*.log; do echo "$(basename $f .log) cases=$(grep -c '^\[sdpa-correctness\] .* S=' $f) passed=$(grep -c '^\[sdpa-correctness\] .* PASSED$' $f) nonzero-mismatch=$(grep '^\[sdpa-correctness\] .* S=' $f | grep -vc ' mismatches=0/') kernels=$(grep -h '^\[sdpa-kernels\]' $f | sed 's/.*qk=//' | sort -u | wc -l)"; done > $D/$s/sdpa-passes.txt; }
  [[ -d $S/sdpa-old ]] && { cp $S/sdpa-old/rc.csv $D/$s/sdpa-old-rc.csv; python3 $ME/sdpa_records.py $S cand > $D/$s/sdpa-records.txt
    for f in $S/sdpa-old/*.log; do grep -h '^\[sdpa-correctness\]\|^\[sdpa-error\]\|^\[sdpa-kernels\]\|^\[sarc_dev\]' $f | cut -c1-400 > $D/$s/sdpa-$(basename $f .log).txt; done; }
  [[ -d $S/verify ]] && { grep -h '^\[production-diff\]\|total mismatched' $S/verify/pdiff-*.log > $D/$s/pdiff.txt 2>/dev/null
    # every SARC kernel with a required subgroup size named in the logs of verify.sh and of the SDPA passes, joined with the static check
    { echo "kernel,required_subgroup_size,local_size_x,multiple"
      cat $S/verify/*.log $S/verify/*.json $S/sdpa-correctness/*.log $M/stage/${s/sBn/sB}/sdpa-correctness/cand-all-r1.log 2>/dev/null | grep -o 'sarc_[a-z0-9_]*' | sort -u | while read -r k; do
        l=$(grep "^$k," $D/local-size.csv | cut -d, -f3-5 | sort -u | tr '\n' ' '); [[ -n $l ]] && echo "$k,${l% }"; done; } > $D/$s/dispatched-kernels.csv
    grep -h -m3 '^\[sarc_dev\]' $S/verify/*.log 2>/dev/null | sort | uniq -c > $D/$s/sarc_dev-banners.txt; }
done
[[ -f $M/stage/sA-fused1/verify.out ]] && vdiff $OLD/stage/s5-c1/verify.out $M/stage/sA-fused1/verify.out > $D/sA-fused1/verify-diff.txt
[[ -f $M/stage/sBn-noenv/verify.out ]] && vdiff $OLD/stage/s7n-noenv/verify.out $M/stage/sBn-noenv/verify.out > $D/sBn-noenv/verify-diff.txt
pd() { # pd <old session of the campaign> <new session>: the production-diff result lines, log by log
  local n=0 d=0 f x; for f in $M/stage/$2/verify/pdiff-*.log; do n=$((n + 1))
    x=$(diff <(grep '^\[production-diff\]\|total mismatched' $OLD/stage/$1/verify/$(basename $f)) <(grep '^\[production-diff\]\|total mismatched' $f) | grep -c '^[<>]'); d=$((d + x)); done
  echo "production-diff, old $1 against new $2: $n logs, $(cat $M/stage/$2/verify/pdiff-*.log | grep -c '^\[production-diff\] ALL PASSED') ALL PASSED and $(cat $M/stage/$2/verify/pdiff-*.log | grep -c '^\[production-diff\] FAILED') FAILED on the new build ($(cat $OLD/stage/$1/verify/pdiff-*.log | grep -c '^\[production-diff\] ALL PASSED') and $(cat $OLD/stage/$1/verify/pdiff-*.log | grep -c '^\[production-diff\] FAILED') on the old); result lines differing, log by log: $d"; }
[[ -d $M/stage/sA-fused1/verify ]] && pd s5-c1 sA-fused1 > $D/sA-fused1/pdiff-diff.txt
[[ -d $M/stage/sBn-noenv/verify ]] && pd s7n-noenv sBn-noenv > $D/sBn-noenv/pdiff-diff.txt
if [[ -f $M/raw/kernel/rc.csv ]]; then cp $M/raw/kernel/rc.csv $M/raw/kernel/*.json $D/kernel/; python3 $ME/kernel_table.py $M/raw/kernel > $D/kernel/kernel.csv 2> $D/kernel/summary.txt; fi
if [[ -f $M/raw/kernel-decode/rc.csv ]]; then mkdir -p $D/kernel-decode; cp $M/raw/kernel-decode/rc.csv $M/raw/kernel-decode/*.json $D/kernel-decode/; cp $M/jobs/decode*.status $D/kernel-decode/
  python3 $ME/decode_table.py $M/raw/kernel-decode $M/raw/kernel > $D/kernel-decode/decode.txt; fi
# logits: the .bin files (excluded by pull.sh) are compared on the device; here the per-cell sha256 list it wrote.
[[ -f $M/probe/f1-logits.sha256 ]] && cp $M/probe/f1-logits.sha256 $M/probe/f1-logits.txt $D/logits/
for a in $M/probe/broad/f1-*/; do [[ -d $a ]] || continue; n=$(basename $a); cat $a/env.txt > $D/logits/$n.env.txt; for f in $a/*.log; do echo "$(basename $f .log): $(tail -1 $f)"; done > $D/logits/$n.cells.txt; done
du -sh $D; tail -qn1 $D/*/verify-diff.txt $D/*/sdpa-records.txt 2>/dev/null; cat $D/kernel/summary.txt 2>/dev/null; cat $D/logits/f1-logits.txt 2>/dev/null
