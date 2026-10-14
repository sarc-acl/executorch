#!/bin/bash
# test_hold.sh: dry run of the coordinator hold (host.sh coordinator_hold / gpu_begin), CPU only, in a temporary
# artifact directory, so it can run beside the queues: no GPU process, no lock of the real artifact directory,
# no process named like a workload. The "jobs" are subshells that take the pair lock through gpu_begin exactly
# as gl.sh does and then sleep. Poll interval 2 s instead of 60 (XE2_HOLD_POLL).
set -u
T=$(dirname "$(readlink -f "$0")"); W=$(mktemp -d); fails=0; export XE2_ARTIFACTS=$W XE2_HOLD_POLL=2
ok() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1   [$2]"; fails=$((fails + 1)); fi; }
job() { ( export XE2_CARD=$1; . $T/host.sh; gpu_begin $2 "job $3"; echo start >> $W/$3; sleep $4; echo end >> $W/$3 ) & }
job 0 shared before 1; wait; ok "without HOLD a job runs and no HELD is written" '[[ -s $W/before && ! -e $W/HELD ]]'
job 1 shared running 6; sleep 1; touch $W/HOLD                       # a job is running when HOLD appears
job 0 shared next0 1; job 1 shared next1 1; job 0 excl nextx 1; sleep 3
ok "the running job is not killed" 'grep -q start $W/running && ! grep -q end $W/running'
ok "no HELD while a job still runs on the other card" '[[ ! -e $W/HELD ]]'
ok "nothing new started" '[[ ! -e $W/next0 && ! -e $W/next1 && ! -e $W/nextx ]]'
sleep 9
ok "the running job finished normally" 'grep -q end $W/running'
ok "HELD written once everything is idle, one line per waiting unit" '[[ $(grep -c "^HELD 20..-..-..T..:..:..Z card[01] job next" $W/HELD) == 3 ]]'
ok "still nothing started" '[[ ! -e $W/next0 && ! -e $W/next1 && ! -e $W/nextx ]]'
job 0 shared late 1; sleep 7; ok "a unit arriving during the hold waits and reports" '[[ ! -e $W/late ]] && grep -q "job late" $W/HELD'
( . $T/host.sh; pair_lock excl; export XE2_PAIR_HELD; ( unset XE2_PAIR_OWN; gpu_begin excl "job nested"; echo start >> $W/nested ) & sleep 3   # a child process: the lock is its parent's
  ok "a unit inside a tool that holds the pair lock waits and reports too" '[[ ! -e $W/nested ]] && grep -q "job nested" $W/HELD'; wait; exit $fails ) & n=$!
sleep 5; cat $W/HELD; rm $W/HOLD; wait
ok "after HOLD is removed every unit ran" 'for f in next0 next1 nextx late nested; do grep -q start $W/$f || exit 1; done'
ok "and HELD is gone" '[[ ! -e $W/HELD ]]'
wait $n; fails=$((fails + $?))
[[ $fails == 0 ]] && { echo "TEST_HOLD_PASS"; rm -rf $W; exit 0; }; echo "TEST_HOLD_FAIL ($fails), evidence in $W"; exit 1
