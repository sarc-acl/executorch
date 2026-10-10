#!/bin/bash
# f1-chain.sh: device side. Everything of task f1-orin that runs on the Orin, in order, with the fused-port
# campaign's tools unchanged (mk-tools.sh: only the two root constants of common.sh differ). old = build topic4
# (the campaign's final build, bundle copied), new = build f1 (0f038a736). A step that fails is recorded and the
# chain goes on; device loss (70), a busy lock (75) or a foreign GPU process (76) end it. Nothing is tuned.
#   2. flag proof (the validation layer is not installed): build f1dbg = f1 + an uncommitted debug print of the
#      stage flags; one linear suite and one SDPA tier with the final configuration, the same with no environment
#   3. session A (orin-fused1): the campaign's gate_sdpa.sh (SDPA tiers, verify.sh against s0-parent-verify,
#      timed session old against new, traces); one SDPA pass per tier of the old and the new build
#   4. session B (no environment): noenv_verify.sh against s0n-noenv; SDPA tiers all / extended / full x 12;
#      timed.sh old against new; one SDPA pass per tier of the old and the new build
#   5. kernel level, old / new interleaved, 3 processes per arm, both environments
#   6. 41-prompt last-position logits (logits_dump) of old and new, both configurations
cd "$(dirname "$0")"; source ./common.sh
OLD=topic4; NEW=f1; U=ET_VK_SARC_UNVERIFIED=1; C1="$U ET_VK_SARC_DEV_PROFILE=orin-fused1"
ST=$A/chain.status; say() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
fatal() { [[ $1 == 70 || $1 == 75 || $1 == 76 ]] && { say "CHAIN_STOPPED rc=$1 ($2)"; exit $1; }; return 0; }
step() { echo "== $(date -u +%FT%TZ) $*"; "$@"; local rc=$?; say "rc=$rc $*"; fatal $rc "$1"; return 0; }
want() { [[ " ${CHAIN_STEPS:-2 3 4 5 6} " == *" $1 "* ]]; }
mb() { echo $A/build/$1/bundle/test_llama_microbench; }
run() { local b=$1 e=$2; shift 2; env $e LD_LIBRARY_PATH=$A/build/$b/bundle ./gl.sh $(mb $b) "$@"; }   # run <build> "<env>" <args>
say "chain start, steps ${CHAIN_STEPS:-2 3 4 5 6}, kernel $(uname -r), $(nvpmodel -q 2>/dev/null | head -1)"
need $(mb $OLD) $(mb $NEW) $A/stage/s0-parent-verify/verify.out $A/stage/s0n-noenv/verify.out

if want 2; then O=$A/raw/flagproof; mkdir -p $O; need $(mb f1dbg)
  for e in fused1 noenv; do E=""; [[ $e == fused1 ]] && E=$C1
    cool_start 120; run f1dbg "$E" --linear --regime=prefill --scheme=4w --skip-correctness > $O/$e-linear.log 2>&1; r1=$?; fatal $r1 flagproof
    cool_start 120; run f1dbg "$E" --sdpa-correctness-only --sdpa-tier=all > $O/$e-sdpa.log 2>&1; r2=$?; fatal $r2 flagproof
    say "step 2 flag proof $e: linear rc=$r1 sdpa rc=$r2; [f1-flagprint] lines $(cat $O/$e-linear.log $O/$e-sdpa.log | grep -c '^\[f1-flagprint\]')"
  done
fi

sdpa_old() { # sdpa_old <session> "<env>": one pass per tier with the OLD and the NEW test binary, for the record comparison
  local O=$A/stage/$1/sdpa-old b tier; mkdir -p $O
  for tier in all extended full peaked; do for b in old:$OLD new:$NEW; do cool_start 120
    run ${b#*:} "$2" --sdpa-correctness-only --sdpa-tier=$tier > $O/${b%:*}-$tier.log 2>&1; rc=$?; echo "${b%:*},$tier,1,$rc" >> $O/rc.csv; fatal $rc sdpa_old
  done; done; say "sdpa old/new single passes $1: $(tr '\n' ' ' < $O/rc.csv)"; }

if want 3; then S=sA-fused1
  step ./stage.sh $S $OLD "$C1" $NEW "$C1" "F1: old build topic4 against new build f1 (full-subgroups flag), both with orin-fused1"
  step ./gate_sdpa.sh $S "$C1"; say "step 3 gate $S: $(cat $A/stage/$S/gate.done 2>/dev/null) | $(grep -h geomean $A/stage/$S/raw/summary.csv 2>/dev/null)"
  sdpa_old $S "$C1"
fi

if want 4; then S=sB-noenv; D=$A/stage/$S; export PARENT_CTL_NAME=s0n-noenv
  step ./noenv_verify.sh $NEW sBn-noenv; say "step 4 noenv verify: $(cat $A/stage/sBn-noenv/gate.done 2>/dev/null)"
  step ./stage.sh $S $OLD "" $NEW "" "F1: old build topic4 against new build f1 (full-subgroups flag), both with no environment (the released kernels)"
  O=$D/sdpa-correctness; mkdir -p $O
  for tier in all extended full; do for i in $(seq 1 12); do cool_start 300
    run $NEW "" --sdpa-correctness-only --sdpa-tier=$tier > $O/cand-$tier-r$i.log 2>&1; rc=$?; echo "cand $tier r$i rc=$rc" >> $O/summary.txt; fatal $rc "sdpa $tier r$i"
  done; done
  python3 gate_check.py sdpa $O $D/cand/env > $D/sdpa-check.txt 2>&1; say "step 4 sdpa passes: nonzero rc $(grep -vc 'rc=0$' $O/summary.txt) of $(wc -l < $O/summary.txt); gate_check sdpa rc=$? $(tail -1 $D/sdpa-check.txt | cut -c1-120)"
  step ./timed.sh $S; say "step 4 timed $S: $(cat $D/gate.done 2>/dev/null) | $(grep -h geomean $D/raw/summary.csv 2>/dev/null)"
  sdpa_old $S ""; unset PARENT_CTL_NAME
fi

if want 5; then O=$A/raw/kernel; mkdir -p $O
  for e in fused1 noenv; do E=""; [[ $e == fused1 ]] && E=$C1
    for r in 1 2 3; do
      if (( r % 2 )); then order="$OLD $NEW"; else order="$NEW $OLD"; fi
      for b in $order; do
        for q in 4w 8da4w; do cool_start 120
          run $b "$E" --linear --regime=prefill --scheme=$q --skip-correctness --json-out=$O/$e-$b-linear-$q-r$r.json > $O/$e-$b-linear-$q-r$r.log 2>&1
          rc=$?; echo "$e,$b,linear-$q,$r,$rc,$(date -u +%FT%TZ)" >> $O/rc.csv; fatal $rc kernel; done
        cool_start 120
        run $b "$E" --sdpa --json-out=$O/$e-$b-sdpa-r$r.json > $O/$e-$b-sdpa-r$r.log 2>&1
        rc=$?; echo "$e,$b,sdpa,$r,$rc,$(date -u +%FT%TZ)" >> $O/rc.csv; fatal $rc kernel
      done
    done
  done
  say "step 5 kernel runs: $(wc -l < $O/rc.csv), nonzero rc: $(awk -F, '$5 != 0' $O/rc.csv | wc -l)"
fi

if want 6; then
  step ./probe_run.sh $OLD f1-old-fused1 $C1
  step ./probe_run.sh $NEW f1-new-fused1 $C1
  step ./probe_run.sh $OLD f1-old-noenv
  step ./probe_run.sh $NEW f1-new-noenv
  for e in fused1 noenv; do n=0; d=0
    for f in $A/probe/broad/f1-old-$e/*.bin; do n=$((n + 1)); cmp -s $f ${f/f1-old-/f1-new-} || d=$((d + 1)); done
    say "step 6 logits $e: $n cell files of the old build (41 prompts each), $d differ from the new build's (cmp)"
  done
fi
say "CHAIN_DONE"
