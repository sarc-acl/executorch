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
    for q in 4w 8da4w; do grep -o '"kernel": "[^"]*"' $S/verify/linear-$q.json 2>/dev/null | sort | uniq -c; done > $D/$s/verify-linear-kernels.txt
    grep -h -m3 '^\[sarc_dev\]' $S/verify/*.log 2>/dev/null | sort | uniq -c > $D/$s/sarc_dev-banners.txt; }
done
[[ -f $M/stage/sA-fused1/verify.out ]] && vdiff $OLD/stage/s5-c1/verify.out $M/stage/sA-fused1/verify.out > $D/sA-fused1/verify-diff.txt
[[ -f $M/stage/sBn-noenv/verify.out ]] && vdiff $OLD/stage/s7n-noenv/verify.out $M/stage/sBn-noenv/verify.out > $D/sBn-noenv/verify-diff.txt
if [[ -f $M/raw/kernel/rc.csv ]]; then cp $M/raw/kernel/rc.csv $M/raw/kernel/*.json $D/kernel/; python3 $ME/kernel_table.py $M/raw/kernel > $D/kernel/kernel.csv 2> $D/kernel/summary.txt; fi
# logits: the .bin files (excluded by pull.sh) are compared on the device; here the per-cell sha256 list it wrote.
[[ -f $M/probe/f1-logits.sha256 ]] && cp $M/probe/f1-logits.sha256 $M/probe/f1-logits.txt $D/logits/
for a in $M/probe/broad/f1-*/; do [[ -d $a ]] || continue; n=$(basename $a); cat $a/env.txt > $D/logits/$n.env.txt; for f in $a/*.log; do echo "$(basename $f .log): $(tail -1 $f)"; done > $D/logits/$n.cells.txt; done
du -sh $D; tail -qn1 $D/*/verify-diff.txt $D/*/sdpa-records.txt 2>/dev/null; cat $D/kernel/summary.txt 2>/dev/null; cat $D/logits/f1-logits.txt 2>/dev/null
