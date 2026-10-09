#!/bin/bash
# chain.sh: everything of task f1-b580 that runs on the card, in order, with the fused-port campaign's tools
# unchanged (openspec/changes/sarc-1.5-b580-fused-port/tools; they resolve the artifact directory from this
# working copy: /mnt/linux-share/hmz-campaigns/f1-b580/.artifacts). old = build topic7 (copied from the campaign),
# new = build f1. Every step runs even after an earlier failure; nothing is tuned. Ends CHAIN_DONE.
#   0. copies: build topic7 (+traced, probe), the calibration files, the parent snapshots; probe of f1
#   1. SPIR-V identity topic7 / f1
#   2. validation layer, old and new: one linear suite and one SDPA tier under b580-fused1
#   3. session A (b580-fused1): the campaign's gate_sdpa.sh, 7 repeats; 1 SDPA pass per tier of the old build
#   4. session B (no environment): the steps of gate.sh with the no-environment snapshot of topic7 as control
#   5. kernel level, old / new interleaved, 3 processes per arm, both environments
#   6. logits of old and new (tools/logits_probe), both sessions
set -uo pipefail
ME=$(dirname "$(readlink -f "$0")"); CT=$(cd $ME/../../sarc-1.5-b580-fused-port/tools && pwd)
. $CT/host.sh
OLD=/mnt/linux-share/hmz-campaigns/b580-fused/.artifacts
export B580_PYTHON=$OLD/venv/bin/python
ST=$A/logs/chain.status; mkdir -p $A/logs $A/raw; say() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
U="ET_VK_SARC_UNVERIFIED=1"; C1="$U ET_VK_SARC_DEV_PROFILE=b580-fused1"
step() { [[ " ${CHAIN_STEPS:-0 1 2 3 4 5 6} " == *" $1 "* ]]; }
say "chain start, tools $(git -C $ET rev-parse --short HEAD), kernel $(uname -r), steps ${CHAIN_STEPS:-all}"

if step 0; then
  grep -qx BUILD_BOTH_OK $A/build/f1.src.txt || { say "CHAIN_STOPPED build f1 is not BUILD_BOTH_OK"; exit 1; }
  if [[ ! -e $A/build/topic7.src.txt ]]; then
    cp -a $OLD/build/topic7 $OLD/build/topic7-traced $A/build/ && cp -a $OLD/build/topic7.src.txt $OLD/build/topic7.golden.txt $A/build/ || { say "CHAIN_STOPPED copy of topic7 failed"; exit 1; }
  fi
  for f in llama/examples/models/llama/llama_main tests/test_llama_microbench probe/logits_probe; do cmp $OLD/build/topic7/$f $A/build/topic7/$f || { say "CHAIN_STOPPED topic7 copy differs: $f"; exit 1; }; done
  cp -a $OLD/clkmin_mhz $OLD/busymax_pct $OLD/idle_temp_mc $A/
  for s in s0-parent-verify s0-topic7-noenv; do [[ -e $A/stage/$s ]] || cp -a $OLD/stage/$s $A/stage/; done
  [[ -x $A/build/f1/probe/logits_probe ]] || $CT/build-probe.sh f1 > $A/logs/build-probe-f1.out 2>&1
  say "step 0 copies done; calibration clkmin=$(<$A/clkmin_mhz) busymax=$(<$A/busymax_pct) idle=$(<$A/idle_temp_mc); probe f1: $(tail -1 $A/logs/build-probe-f1.out)"
fi

if step 1; then
  ( cd $A/build/topic7 && find . -name '*.spv' | sort | xargs sha256sum ) > $A/raw/spv-topic7.sha
  ( cd $A/build/f1 && find . -name '*.spv' | sort | xargs sha256sum ) > $A/raw/spv-f1.sha
  { echo "topic7: $(wc -l < $A/raw/spv-topic7.sha) .spv files, f1: $(wc -l < $A/raw/spv-f1.sha)"
    echo "differences (diff of the two sorted sha256 lists): $(diff $A/raw/spv-topic7.sha $A/raw/spv-f1.sha | grep -c '^[<>]')"
    diff $A/raw/spv-topic7.sha $A/raw/spv-f1.sha | head -40
    echo "sha256 of the list: $(sha256sum < $A/raw/spv-topic7.sha | cut -c1-64) / $(sha256sum < $A/raw/spv-f1.sha | cut -c1-64)"; } > $A/raw/spirv-identity.txt
  say "step 1 $(sed -n 2p $A/raw/spirv-identity.txt)"
fi

if step 2; then
  O=$A/raw/validation; mkdir -p $O
  { vulkaninfo --summary 2>/dev/null | grep -A8 'VK_LAYER_KHRONOS_validation' | head -4; rpm -q vulkan-validation-layers 2>/dev/null; } > $O/layer.txt
  for b in topic7 f1; do B=$A/build/$b/tests/test_llama_microbench
    cool_start
    env $C1 VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation $CT/gl.sh $B --linear --regime=prefill --scheme=4w --storage=texture3d --skip-correctness > $O/$b-linear.log 2>&1; r1=$?
    env $C1 VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation $CT/gl.sh $B --sdpa-correctness-only --sdpa-tier=all > $O/$b-sdpa.log 2>&1; r2=$?
    say "step 2 validation $b linear rc=$r1 sdpa rc=$r2 VUID-10770 lines: $(cat $O/$b-linear.log $O/$b-sdpa.log | grep -c 'OpTypeCooperativeMatrixKHR-10770') any VUID lines: $(cat $O/$b-linear.log $O/$b-sdpa.log | grep -c 'VUID-')"
  done
fi

sdpa_old() { # sdpa_old <session> "<env>": one pass per tier (and peaked) with the OLD build's test binary, for the record comparison
  local O=$A/stage/$1/sdpa-old; mkdir -p $O
  for tier in all extended full peaked; do cool_start
    env $2 $CT/gl.sh $A/build/topic7/tests/test_llama_microbench --sdpa-correctness-only --sdpa-tier=$tier > $O/old-$tier.log 2>&1; echo "old,$tier,1,$?" >> $O/rc.csv
    env $2 $CT/gl.sh $A/build/f1/tests/test_llama_microbench --sdpa-correctness-only --sdpa-tier=$tier > $O/new-$tier.log 2>&1; echo "new,$tier,1,$?" >> $O/rc.csv
  done; }

if step 3; then
  S=sA-fused1
  $CT/stage.sh $S topic7 "$C1" f1 "$C1" "F1: old build topic7 against new build f1 (full-subgroups flag), both with b580-fused1" > $A/logs/$S.stage.out 2>&1 || say "staging $S failed"
  B580_REPS=7 $CT/gate_sdpa.sh $S > $A/logs/gate-$S.out 2>&1; say "step 3 gate $S rc=$? $(cat $A/stage/$S/gate.done 2>/dev/null) | $(grep -h geomean $A/stage/$S/raw/summary.csv | cut -c1-70)"
  sdpa_old $S "$C1"; say "step 3 sdpa old/new single passes: $(tr '\n' ' ' < $A/stage/$S/sdpa-old/rc.csv)"
fi

if step 4; then
  S=sB-noenv; D=$A/stage/$S
  $CT/stage.sh $S topic7 "" f1 "" "F1: old build topic7 against new build f1 (full-subgroups flag), both with no environment (the released kernels)" > $A/logs/$S.stage.out 2>&1 || say "staging $S failed"
  gpu_shared || exit 75
  $CT/sdpa_passes.sh $S table 12 "" > $D/sdpa.out 2>&1; say "step 4 sdpa passes rc=$?"
  cool_start
  guarded $D/verify.others $ET/sarc/tools/verify.sh --dir $D --lock $LOCK --models 1b,3b,8b --schemes 4w,8da4w --pdiff --out verify > $D/verify.out 2>&1
  rc=$?; echo "VERIFY_DONE rc=$rc" >> $D/verify.out; say "step 4 verify rc=$rc"
  $CT/session.sh $S --reps 7 > $D/e2e5.out 2>&1; say "step 4 session rc=$? $(tail -1 $D/e2e5.out | cut -c1-120)"
  $CT/trace.sh $S 1b,3b,8b 4w,8da4w "parent cand" > $D/trace.out 2>&1; say "step 4 trace rc=$?"
  python3 $CT/summarize.py $D/raw > $D/raw/summary.csv 2>&1
  python3 $CT/gate_check.py $D --parent $A/stage/s0-topic7-noenv/verify.out > $D/gate.txt 2>&1; rc=$?; tail -1 $D/gate.txt > $D/gate.done
  python3 $CT/gate_check.py $D --control > $D/gate-control.txt 2>&1
  say "step 4 gate $S rc=$rc $(cat $D/gate.done) | control: $(tail -1 $D/gate-control.txt) | $(grep -h geomean $D/raw/summary.csv | cut -c1-70)"
  sdpa_old $S ""; say "step 4 sdpa old/new single passes: $(tr '\n' ' ' < $D/sdpa-old/rc.csv)"
fi

if step 5; then
  O=$A/raw/kernel; mkdir -p $O
  for e in fused1 noenv; do E=""; [[ $e == fused1 ]] && E=$C1
    for r in 1 2 3; do
      if (( r % 2 )); then order="topic7 f1"; else order="f1 topic7"; fi
      for b in $order; do B=$A/build/$b/tests/test_llama_microbench
        for q in 4w 8da4w; do cool_start
          env $E $CT/gl.sh $B --linear --regime=prefill --scheme=$q --storage=texture3d --skip-correctness --json-out=$O/$e-$b-linear-$q-r$r.json > $O/$e-$b-linear-$q-r$r.log 2>&1
          echo "$e,$b,linear-$q,$r,$?,$(date -u +%FT%TZ)" >> $O/rc.csv; done
        cool_start
        env $E $CT/gl.sh $B --sdpa --json-out=$O/$e-$b-sdpa-r$r.json > $O/$e-$b-sdpa-r$r.log 2>&1
        echo "$e,$b,sdpa,$r,$?,$(date -u +%FT%TZ)" >> $O/rc.csv
      done
    done
  done
  say "step 5 kernel runs: $(wc -l < $O/rc.csv), nonzero rc: $(awk -F, '$5 != 0' $O/rc.csv | wc -l)"
fi

if step 6; then
  for S in sA-fused1 sB-noenv; do
    B580_PROBE_ARMS="P C" $CT/probe.sh $S > $A/logs/probe-$S.out 2>&1; rc=$?
    n=0; d=0; for f in $A/stage/$S/probe/*/P/*.f32; do n=$((n + 1)); cmp -s $f ${f/\/P\//\/C\/} || d=$((d + 1)); done
    echo "$S: $n logits vectors of the old build, $d differ from the new build's (cmp, byte for byte)" | tee $A/stage/$S/probe/identity.txt
    say "step 6 probe $S rc=$rc: $n vectors, $d differ"
  done
fi
say "CHAIN_DONE"
