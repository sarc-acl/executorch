#!/bin/bash
# collect-head4.sh: copy the small evidence files of the replacement build head4 (chain27.sh, chain28.sh) from
# <artifacts 10-08> into results/780m/. Raw logs, clock samples, dumps, ETDumps and binaries stay in the artifacts.
set -uo pipefail
A=$HOME/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-08; C=$HOME/hmz-sarc/executorch/openspec/changes/sarc-1.5-780m-prefill-refine
R=$C/results/780m; H=$R/round3/head4; W=$A/raw/head4; mkdir -p $H/spirv $H/bitwise
cp -f $W/export-check.txt $W/submodules-head4-vs-head3.txt $W/binaries-sha256.txt $H/
sed -i "s|$A/||g" $H/binaries-sha256.txt
sed "s|$A/||g" $A/src/head4/MANIFEST > $H/build-head4.EXPORT-MANIFEST; cp -f $A/build/head4.src.txt $H/build-head4.src.txt
cp -f $A/logs/export-head4.out $H/export-head4.out; cp -f $W/spirv/*.txt $H/spirv/
cp -f $A/logs/chain27.status $A/logs/chain28.status $R/round3/
{ echo "# SDPA output of every correctness case, c11 with the fused3 pair (ET_VK_SARC_780M_SDPA_FUSED) against c11 as committed (fused3sb), byte for byte."
  echo "# build head4 (c639d4760, export from object stores), test binary of stage r3a-fused3sb-head4; tiers all, extended, peaked, full (21 cases) and fused (5 cases); dumps in <artifacts 10-08>/raw/head4/bitwise/dump-{fused3,fused3sb}."
  cat $W/bitwise/compare.txt; } > $H/bitwise/bitwise-fused3sb.txt
cp -f $W/bitwise/compare-head3.txt $H/bitwise/
for a in fused3 fused3sb; do for t in all extended peaked full fused; do grep -h '\[sdpa-error\]' $W/bitwise/$a-$t.log | sed "s/^/$a $t /"; done; done > $H/bitwise/sdpa-error.txt
{ for a in fused3 fused3sb; do for t in all extended peaked full fused; do f=$W/bitwise/$a-$t.log; echo "$a $t $(tail -1 $f) passed=$(grep -c PASSED $f) failed=$(grep -c FAILED $f)"; done; done; } > $H/bitwise/runs.txt
{ echo "arm,model,rep,op_mean_us_per_layer,op_stdev_us  # test binary of stage r3a-fused3sb-head4 (build head4, c639d4760), --sdpa, ET_VK_SDPA_PERF_RUNS=40,10; RESULT rows prefill,total of the coopmat path = copy pass + fused kernel; fused3 = c11 with ET_VK_SARC_780M_SDPA_FUSED naming the fused3 pair, fused3sb = c11 as committed"
  for a in fused3 fused3sb; do for i in 1 2 3; do grep -h '^RESULT,sdpa,.*,prefill,total,.*,coopmat$' $W/kernel-time/$a-r$i.log | awk -F, -v a=$a -v i=$i '{printf "%s,%s,%s,%.1f,%.1f\n", a, $3, i, $9, $10}'; done; done; } > $R/fused/kernel-time-steady-r3-fused3sb-head4.csv
for n in r3a-fused3sb-head4 r3b-final-dev15-head4; do s=$A/stage/$n; D=$R/sessions/$n; mkdir -p $D/verify $D/sdpa-correctness
  cp -f $s/STAGE.md $s/raw/runs.csv $s/raw/env.txt $s/raw/nexttoken.csv $s/prestart.txt $s/verify.out $s/verify-residency.txt $D/
  sed -i "s|$A/||g" $D/STAGE.md $D/env.txt
  python3 $A/tools/summarize.py $s/raw > $D/summary.csv
  cp -f $s/verify/env.txt $s/verify/correctness.log $D/verify/
  for q in 4w 8da4w; do grep -E '^(linear|baseline) |geomean|unexpected|confirmed|crashed' $s/verify/linear-$q.log > $D/verify/linear-$q.txt; done
  [[ -d $s/trace/report/evidence/trace ]] && { mkdir -p $D/trace; cp -f $s/trace/report/evidence/trace/*.csv $s/trace/raw/780m/trace2/residency.txt $D/trace/; }
  cp -f $s/sdpa-correctness/summary.txt $D/sdpa-correctness/
  { echo "# distinct [sdpa-kernels] lines over the passes of each tier, candidate environment; count = cases x passes"
    for t in all extended full; do cat $s/sdpa-correctness/cand-$t-r*.log | grep -o '\[sdpa-kernels\].*' | sed 's/\[sdpa-kernels\] [^ ]* /[sdpa-kernels] <case> /' | sort | uniq -c | sed "s/^/$t /"; done
    echo "# mismatches: $(cat $s/sdpa-correctness/cand-*.log | grep -o 'mismatches=[0-9]*' | sort | uniq -c | tr '\n' ' ')"
    ls $s/sdpa-correctness/table-*.log > /dev/null 2>&1 && { echo "# control, table kernels, 1 pass per tier"
      for t in all extended full; do grep -o '\[sdpa-kernels\].*' $s/sdpa-correctness/table-$t-r1.log | sed 's/\[sdpa-kernels\] [^ ]* /[sdpa-kernels] <case> /' | sort | uniq -c | sed "s/^/$t /"; done
      echo "# mismatches: $(cat $s/sdpa-correctness/table-*.log | grep -o 'mismatches=[0-9]*' | sort | uniq -c | tr '\n' ' ')"; }
  } > $D/sdpa-correctness/kernels.txt
done
{ echo "# Dispatched kernel names, build head4 (c639d4760, export from object stores) against candidate 11's gate (build head2, a8fffa5ea)."
  echo "# 1. verify.sh kernel lines (linear 4w / 8da4w)"
  for x in "candidate 11's gate:$R/sessions/c11-dq-refine11/verify.out" "head4, 780m-refine3 + c11:$R/sessions/r3a-fused3sb-head4/verify.out" "head4, ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=780m-final, ET_VK_SARC_780M_PROFILE unset:$R/sessions/r3b-final-dev15-head4/verify.out"; do
    echo "## ${x%%:*} (${x#*:})" | sed "s|$R/|results/780m/|"; grep '^linear ' "${x#*:}"; done
  strip() { sed -E 's/(tok_s=)[0-9.]+/\1X/g' "$1"; }
  G=$R/sessions/c11-dq-refine11/verify.out
  for n in r3a-fused3sb-head4 r3b-final-dev15-head4; do
    echo "## whole verify.out of $n against candidate 11's gate, tok/s figures set aside: $(diff <(strip $G) <(strip $R/sessions/$n/verify.out) | grep -c '^[<>]') differing lines of $(wc -l < $R/sessions/$n/verify.out)"
    echo "## whole verify.out of $n against the same stage on head3: $(diff <(strip $R/sessions/${n%-head4}/verify.out) <(strip $R/sessions/$n/verify.out) | grep -c '^[<>]') differing lines"
    echo "## whole verify.out of $n against the parent snapshot s0-parent-verify: lines $(diff <(strip $HOME/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-03/stage/s0-parent-verify/verify.out) <(strip $R/sessions/$n/verify.out) | grep -E '^[0-9]' | tr '\n' ' ')differ"
  done
  echo "# 2. SDPA kernels ([sdpa-kernels] lines of --sdpa-correctness-only, tier all; qk / softmax / av are '?' when the fused node serves the case); <artifacts 10-08>/raw/head4/smoke/"
  for f in none refine3 c10 c11-fused3 c11 final final-no-unverified final-with-c10 c11-nofused final-nofused; do l=$W/smoke/$f.log
    echo "## $f: $(tail -1 $l), $(grep -c PASSED $l) passed, $(grep -c FAILED $l) failed"; grep -o '\[sdpa-kernels\].*' $l | sed 's/\[sdpa-kernels\] [^ ]* //' | sort | uniq -c; done
} > $H/dispatch.txt
cp -f $A/tools/*.sh $A/tools/*.py $C/tools/ 2>/dev/null; git -C $C status --short . | head -40
