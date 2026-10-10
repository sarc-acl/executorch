#!/bin/bash
# run_t3.sh pdiff <token...> | time <rounds> <token...>   (H1 test 3; binary: .artifacts/build/exp1)
#   token "base" = the incumbent configuration (ET_VK_SARC_DEV_PROFILE=b580-fused1); any other token = an
#   experiment kernel (ET_VK_SARC_EXP=exp-h1 ET_VK_SARC_DQ8CA_VARIANT=<token>). ET_VK_SARC_UNVERIFIED=1 always.
# pdiff: data/pdiff/<token>-<model>-<zp|nzp>.log = --production-diff on the 4 shapes of each model, with zero and
#        with non-zero activation zero points. Correctness evidence only; no timing is read from these.
# time:  data/t3/<token>-r<round>.json = one process over the 12 prefill shapes (3 warm-up + 5 timed iterations per
#        shape); every round runs every token once, order rotated per round; the card cools before each process
#        (package temperature no longer falling over 30 s, 5 min at most). A process with a foreign GPU workload
#        (76), a mean foreign desktop share above 5.0 % or no readable share sample (77) is moved to
#        data/t3/superseded/<reason>/ and repeated, at most 3 attempts. Resumable: an accepted .json is kept.
. "$(dirname "$(readlink -f "$0")")/guard.sh"; BOX=$(cd $ROOT/../../../.. && pwd); BIN=$BOX/.artifacts/build/exp1/tests/test_llama_microbench
export TMPDIR=$BOX/.tmp ETVK_DEVICE_INDEX=0 ET_VK_SARC_UNVERIFIED=1; mkdir -p $TMPDIR; unset ET_VK_SARC_DEV_PROFILE ET_VK_SARC_DQ8CA_VARIANT ET_VK_SARC_EXP
HW=$(echo /sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon*); gtemp() { cat $HW/temp2_input; }
cool() { local t0=$SECONDS a; while (( SECONDS - t0 < 300 )); do a=$(gtemp); sleep 30; (( $(gtemp) >= a )) && break; done; }
cfg() { if [[ $1 == base ]]; then echo "ET_VK_SARC_DEV_PROFILE=b580-fused1"; else echo "ET_VK_SARC_EXP=exp-h1 ET_VK_SARC_DQ8CA_VARIANT=$1"; fi; }
MODE=${1:?mode}; shift; S=$LOGS/t3-$MODE.status; echo "$(date -u +%FT%TZ) RUNNING pid $$ $MODE $* $(sha256sum $BIN | cut -c1-16)" >> $S
if [[ $MODE == pdiff ]]; then O=$ROOT/data/pdiff; mkdir -p $O
  for t in "$@"; do for m in llama-3.2-1b llama-3.2-3b llama-3.1-8b; do for z in zp nzp; do
    f=$O/$t-$m-$z; [[ -s $f.rc ]] && continue
    gpu_job pdiff-$t-$m-$z env $(cfg $t) timeout 1800 $BIN --production-diff --production-diff-op=8da4w --production-diff-storage=texture3d \
      --production-diff-model=$m $([[ $z == nzp ]] && echo --production-diff-nonzero-zp) > $f.log 2>&1; echo $? > $f.rc
    echo "$(date -u +%FT%TZ) $t $m $z rc=$(<$f.rc) $(grep -c 'correctness=PASSED' $f.log) passed; $(tail -1 $f.log | cut -c1-90)" >> $S
  done; done; done
else R=${1:?rounds}; shift; O=$ROOT/data/t3; mkdir -p $O; T=("$@"); n=${#T[@]}
  for ((r = 1; r <= R; r++)); do for ((i = 0; i < n; i++)); do t=${T[$(( (i + r - 1) % n ))]}; f=$O/$t-r$r
    [[ -s $f.json && -s $f.rc ]] && continue
    for try in 1 2 3; do
      cool; echo "$(date -u +%FT%TZ) temp_mc=$(gtemp)" > $f.started
      gpu_job_busy t3-$t-r$r env $(cfg $t) timeout 900 $BIN --linear --regime=prefill --scheme=8da4w --storage=texture3d --skip-correctness \
        --json-out=$f.json > $f.log 2>&1; rc=$?
      echo "$(date -u +%FT%TZ) $t r$r try $try rc=$rc share=$(share_mean t3-$t-r$r) temp_mc=$(gtemp)" >> $S
      if [[ $rc == 76 || $rc == 77 || $rc == 75 ]]; then d=$O/superseded/$([[ $rc == 77 ]] && echo busy || echo foreign)/$t-r$r-try$try-$(date -u +%H%M%S); mkdir -p $d
        mv $f.* $LOGS/t3-$t-r$r.* $d/ 2>/dev/null; continue; fi
      echo $rc > $f.rc; break
    done
  done; done
fi
echo "$(date -u +%FT%TZ) DONE" >> $S
