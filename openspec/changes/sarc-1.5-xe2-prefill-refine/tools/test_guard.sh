#!/bin/bash
# test_guard.sh: CPU-only regression of the foreign-GPU-process handling. No model runs and no GPU work is
# submitted: llama_main and test_llama_microbench are shell stubs, the artifact directory, HOME (hence the
# gpu-lab lock) and the calibration live in a temporary directory, and the "foreign process" is a python3
# process with an unfamiliar name that only holds the b70-0 render node open (a DRM client, no job).
#   1. guard:  an unfamiliar DRM client is detected; our own descendants and launcher shells are not; a job is
#              stopped within seconds of a foreign process appearing, never completes, and the foreign process
#              is left running. Two cards: a DRM client of the second card is foreign for a job on the first;
#              this campaign's own job on the other card (a registered gl.sh) is not, an unregistered one is;
#              two cheap screens (XE2_SHARED) run side by side, any other job waits for the other card to idle.
#   2. gate A: a foreign process that appears before a timed launch -> e2e5.sh/session.sh exit 76,
#              gate.done = GATE_ABORTED, exit 76, tracing never started.
#   4. screen: with a foreign process present screen.sh launches nothing further, records SCREEN_ABORTED and
#              exits 76.
#   5. parent control: a foreign process during verify.sh -> parent_verify.sh stops it, GATE_ABORTED, exit 76,
#              no gate_check.py verdict.
#   3. gate B: a foreign process that appears during a timed run -> the runner is stopped (its completion
#              marker is absent), runs.csv keeps the row with other_gpu_process, GATE_ABORTED, no tracing.
# Takes about 4 minutes (the session's 60 s idle wait, twice). Exit status 0 only if every assertion holds.
set -u
T=$(dirname "$(readlink -f "$0")"); W=$(mktemp -d); fails=0
ok() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1   [$2]"; fails=$((fails + 1)); fi; }
unset XE2_TOP; export HOME=$W/home XE2_ARTIFACTS=$W/art; mkdir -p $HOME/.cache/gpu-lab $XE2_ARTIFACTS
RN=/dev/dri/$(basename /sys/bus/pci/devices/0000:01:00.0/drm/renderD*)
RN1=/dev/dri/$(basename /sys/bus/pci/devices/0000:02:00.0/drm/renderD*)
foreign() { # an unfamiliar process name holding the render node (of b70-0, or the node given); prints its pid
  rm -f $W/foreign.pid
  python3 -c "import os,sys,time; os.open('${1:-$RN}', os.O_RDWR); sys.stdout.write(str(os.getpid())+'\n'); sys.stdout.flush(); time.sleep(600)" \
    > $W/foreign.pid 2>/dev/null 9>&- & disown; while [[ ! -s $W/foreign.pid ]]; do sleep 0.1; done; cat $W/foreign.pid; }

echo "== 1 guard"
# Each block is a subshell acting as a top-level campaign tool (XE2_TOP = its own pid); the foreign process is
# started by this script, outside that subshell.
( . $T/host.sh; XE2_TOP=$BASHPID
  ok "idle card: nothing foreign" '[[ -z $(gpu_others) ]]'
  guarded $W/g0 bash -c "exec -a llama_main sleep 2"; ok "own job named like a GPU workload runs to completion" "[[ $? == 0 ]]"
  guarded $W/g0 python3 -c "import os,time; os.open('$RN', os.O_RDWR); time.sleep(2)"; ok "own DRM client runs to completion" "[[ $? == 0 ]]"
  exit $fails ); fails=$((fails + $?))
F=$(foreign $RN1)
( . $T/host.sh; XE2_TOP=$BASHPID; ok "DRM client of the second card detected from the first" '[[ $(gpu_others) == *"$F:python3"* ]]'; exit $fails ); fails=$((fails + $?))
( export XE2_CARD=1; . $T/host.sh; XE2_TOP=$BASHPID; ok "and from the second" '[[ $(gpu_others) == *"$F:python3"* ]]'; exit $fails ); fails=$((fails + $?)); kill $F
XE2_CARD=1 XE2_SHARED=1 $T/gl.sh bash -c "exec -a llama_main sleep 5" & g=$!; sleep 2
( . $T/host.sh; XE2_TOP=$BASHPID
  ok "this campaign's job on the other card is not foreign" '[[ -z $(gpu_others) ]]'
  t0=$SECONDS; XE2_TOP= XE2_SHARED=1 $T/gl.sh true; ok "two cheap screens run side by side" "[[ $? == 0 ]] && (( SECONDS - t0 < 2 ))"
  t0=$SECONDS; XE2_TOP= $T/gl.sh true; ok "any other job waits until the other card is idle" "[[ $? == 0 ]] && (( SECONDS - t0 >= 2 ))"
  exit $fails ); fails=$((fails + $?)); wait $g; ok "the other card's job ran to completion" "[[ $? == 0 ]]"
bash -c "exec -a llama_main sleep 5" & g=$!; sleep 1
( . $T/host.sh; XE2_TOP=$BASHPID; ok "an unregistered job named like a GPU workload is foreign" '[[ $(gpu_others) == *"$g:"* ]]'; exit $fails ); fails=$((fails + $?)); wait $g
F=$(foreign)
( . $T/host.sh; XE2_TOP=$BASHPID
  ok "unfamiliar DRM client detected" '[[ $(gpu_others) == *"$F:python3"* ]]'
  guarded $W/g1 bash -c "echo started > $W/j1; sleep 5; echo JOB_COMPLETED >> $W/j1" 2>/dev/null; ok "job not started while a foreign process runs" "[[ $? == 76 && ! -e $W/j1 ]]"
  exit $fails ); fails=$((fails + $?)); kill $F
# An idle monitor: a process named nvtop that only holds the render node (zero engine cycles, zero GPU memory).
cp "$(readlink -f "$(command -v python3)")" $W/nvtop; rm -f $W/mon.pid
$W/nvtop -c "import os,sys,time; os.open('$RN', os.O_RDWR); sys.stdout.write(str(os.getpid())+'\n'); sys.stdout.flush(); time.sleep(600)" > $W/mon.pid 2>/dev/null 9>&- & disown
while [[ ! -s $W/mon.pid ]]; do sleep 0.1; done; MON=$(cat $W/mon.pid)
( . $T/host.sh; XE2_TOP=$BASHPID
  ok "idle nvtop monitor is not a foreign GPU process" '[[ $(gpu_others) != *"$MON:"* ]]'
  ok "idle nvtop monitor is recorded" '[[ $(gpu_monitors) == *"$MON:nvtop"* ]]'
  ok "the same idle DRM client under another name is foreign" '! monitor_idle $$'
  bash -c ': a shell whose command text names llama_main and test_llama_microbench; sleep 4; :' & IS=$!; sleep 0.5
  ok "an inline shell command that only names a workload in its text is not foreign" '[[ $(gpu_others) != *"$IS:"* ]]'; wait $IS
  exit $fails ); fails=$((fails + $?)); kill $MON
( . $T/host.sh; XE2_TOP=$BASHPID; t0=$SECONDS
  guarded $W/g2 bash -c "echo started > $W/j2; bash -c 'sleep 41; echo JOB_COMPLETED >> $W/j2'; echo JOB_COMPLETED >> $W/j2" 2>/dev/null; rc=$?
  ok "job stopped with 76 when a foreign process appears" "[[ $rc == 76 ]]"
  ok "stopped promptly (< 12 s; the job would take 41 s)" "(( SECONDS - t0 < 12 ))"
  sleep 1; ok "job and its descendants never completed" "[[ -s $W/j2 ]] && ! grep -q JOB_COMPLETED $W/j2 && ! ps -eo args | grep -qx 'sleep 41'"
  exit $fails ) & g=$!
sleep 3; F=$(foreign); wait $g; fails=$((fails + $?))
ok "foreign process recorded and left running" "grep -q $F:python3 $W/g2 && kill -0 $F"; kill $F

stage() { # stage <session>: a staged session of stubs, a calibration and a parent control
  local D=$XE2_ARTIFACTS/stage/$1; mkdir -p $D/{parent,cand,parent-traced,cand-traced}
  cat > $D/llama_main <<'S'
#!/bin/bash
# stub runner: the timed arms (run from parent/ or cand/) take 31 s, everything else returns at once
case $(basename "$(dirname "$0")") in parent|cand) sleep 31; echo JOB_COMPLETED ;; esac
S
  printf '#!/bin/bash\nexit 0\n' > $D/test_llama_microbench; : > $D/libllama_runner.so; chmod +x $D/llama_main $D/test_llama_microbench
  for d in parent cand parent-traced cand-traced; do cp $D/llama_main $D/libllama_runner.so $D/$d/; : > $D/$d/env; done
  echo test > $D/STAGE.md; : > $D/prompt_2048.txt; : > $D/prompt_check.txt
  echo 999000 > $XE2_ARTIFACTS/idle_temp_mc; echo 1 > $XE2_ARTIFACTS/clkmin_mhz
  mkdir -p $XE2_ARTIFACTS/stage/s0-parent-verify; echo "VERIFY_DONE rc=0" > $XE2_ARTIFACTS/stage/s0-parent-verify/verify.out; }
gate_case() { # gate_case <session> <file whose appearance triggers the foreign process>
  local D=$XE2_ARTIFACTS/stage/$1 F; stage $1
  bash $T/gate.sh $1 > $W/$1.out 2>&1 & local g=$!
  while [[ ! -e $D/$2 ]] && kill -0 $g 2>/dev/null; do sleep 0.5; done
  F=$(foreign); wait $g; RC=$?; FP=$F; }

echo "== 2 gate A: foreign process before a timed launch"
gate_case ta raw/env.txt; D=$XE2_ARTIFACTS/stage/ta
ok "session aborted before launching" 'grep -q "E2E5_ABORTED other GPU process before" $D/raw/done.txt && [[ $(wc -l < $D/raw/runs.csv) == 1 ]]'
ok "gate exit status 76" '[[ $RC == 76 ]]'
ok "gate.done = GATE_ABORTED" 'grep -q "^GATE_ABORTED foreign GPU process during the timing session" $D/gate.done'
ok "tracing never started" '[[ ! -e $D/trace && ! -e $D/trace.out && ! -e $D/gate.txt ]]'
ok "foreign process left running" 'kill -0 $FP'; kill $FP

echo "== 3 gate B: foreign process during a timed run"
t0=$SECONDS; gate_case tb raw/logs/prefill-1b-4w-parent-r1.clk; D=$XE2_ARTIFACTS/stage/tb
ok "runner stopped, never completed" '! grep -q JOB_COMPLETED $D/raw/logs/prefill-1b-4w-parent-r1.log && ! ps -eo args | grep -qx "sleep 31"'
ok "stopped promptly (session ended < 100 s after start, a full run pair takes 60 s + 2 x 31 s)" '(( SECONDS - t0 < 100 ))'
ok "interrupted run kept as an invalid row" 'grep -q "other_gpu_process" $D/raw/runs.csv && [[ $(wc -l < $D/raw/runs.csv) == 2 ]]'
ok "session status" 'grep -q "E2E5_ABORTED other GPU process during prefill 1b 4w parent r1" $D/raw/done.txt'
ok "gate exit status 76" '[[ $RC == 76 ]]'
ok "gate.done = GATE_ABORTED" 'grep -q "^GATE_ABORTED" $D/gate.done'
ok "tracing never started" '[[ ! -e $D/trace && ! -e $D/trace.out && ! -e $D/gate.txt ]]'
ok "foreign process left running" 'kill -0 $FP'; kill $FP
echo "== 4 screen.sh: foreign process present"
B=$XE2_ARTIFACTS/build/tb/tests; mkdir -p $B; printf '#!/bin/bash\nsleep 2; echo JOB_COMPLETED\n' > $B/test_llama_microbench; chmod +x $B/test_llama_microbench
FP=$(foreign); bash $T/screen.sh scr tb 4w 2 base t1 t2 > $W/scr.out 2>&1; RC=$?; O=$XE2_ARTIFACTS/raw/scr
ok "screen exit status 76" '[[ $RC == 76 ]]'
ok "nothing launched after the first refusal" '[[ $(ls $O/*.log | wc -l) == 1 ]] && ! grep -q JOB_COMPLETED $O/*.log'
ok "screen recorded as aborted, not done" 'grep -q "^SCREEN_ABORTED rc=76 at 4w base r1" $O/env.txt && ! grep -q SCREEN_DONE $O/env.txt'
kill $FP; rm -f $W/foreign.pid

echo "== 5 parent_verify.sh: foreign process during verify.sh"
P=$XE2_ARTIFACTS/build/parent; mkdir -p $P/llama/examples/models/llama $P/llama/lib $P/tests; echo BUILD_BOTH_OK > $XE2_ARTIFACTS/build/parent.src.txt
printf '#!/bin/bash\nexit 0\n' > $P/llama/examples/models/llama/llama_main; : > $P/llama/lib/libllama_runner.so
printf '#!/bin/bash\ncase "$*" in *--correctness-only*) sleep 21; echo JOB_COMPLETED ;; esac\n' > $P/tests/test_llama_microbench
chmod +x $P/llama/examples/models/llama/llama_main $P/tests/test_llama_microbench; : > $W/r1304.txt
rm -rf $XE2_ARTIFACTS/stage/s0-parent-verify; D=$XE2_ARTIFACTS/stage/s0-parent-verify
XE2_UNALIGNED=$W/r1304.txt bash $T/parent_verify.sh > $W/pv.out 2>&1 & g=$!
while [[ ! -e $D/verify/correctness.log ]] && kill -0 $g 2>/dev/null; do sleep 0.5; done
FP=$(foreign); wait $g; RC=$?
ok "control reached verify.sh after the SDPA passes" '[[ $(wc -l < $D/sdpa-correctness/rc.csv) == 3 ]] && ! grep -qv ",0$" $D/sdpa-correctness/rc.csv'
ok "verify.sh stopped" 'grep -q "VERIFY_DONE rc=76" $D/verify.out && ! grep -q JOB_COMPLETED $D/verify/correctness.log && ! ps -eo args | grep -qx "sleep 21"'
ok "parent control exit status 76" '[[ $RC == 76 ]]'
ok "gate.done = GATE_ABORTED, no verdict from gate_check.py" 'grep -q "^GATE_ABORTED foreign GPU process during verify.sh" $D/gate.done && [[ ! -e $D/gate.txt ]]'
ok "foreign process left running" 'kill -0 $FP'; kill $FP

[[ $fails == 0 ]] && { echo "TEST_GUARD_PASS"; rm -rf $W; exit 0; }
echo "TEST_GUARD_FAIL ($fails), evidence kept in $W"; exit 1
