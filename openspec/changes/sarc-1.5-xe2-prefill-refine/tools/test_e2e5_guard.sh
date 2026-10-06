#!/bin/bash
# test_e2e5_guard.sh: regression of the guard of the timing runner THROUGH ITS ENTRY POINT (tools/e2e5.sh, as
# session.sh starts it), CPU only. Everything lives in a temporary directory: the artifact directory
# (XE2_ARTIFACTS, hence HOLD / HELD / run/pair.lock), HOME (hence the gpu-lab lock), a staged session of shell
# stubs and a stub vulkaninfo. The campaign's real artifact directory and the coordinator's real HOLD file are
# never touched. A session counts as "started" when e2e5.sh has written <session>/raw/env.txt, which is the
# first thing it does after the guard; the test stops it before its 60 s idle wait ends, so no stub runner is
# ever launched.
#   1. a coordinator hold (the test's own HOLD) keeps the session from starting, HELD names e2e5.sh, and the
#      session starts when HOLD is removed;
#   2. an occupied pair lock (a cheap screen on the other card) blocks the session until it is released;
#   3. a guard that cannot be acquired stops the runner with status 75 before anything is run;
#   4. the runner's output has no "command not found".
# test_hold.sh covers the functions; this covers the script that uses them (the defect found in s10-final5).
set -u
T=$(dirname "$(readlink -f "$0")"); W=$(mktemp -d); fails=0
ok() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1   [$2]"; fails=$((fails + 1)); fi; }
REAL=${XE2_ARTIFACTS:-$(cd $T/../../../../.. && pwd)/.artifacts}; before=$(ls -la --time-style=full-iso $REAL/HOLD $REAL/HELD $REAL/run/pair.lock 2>&1)
export HOME=$W/home XE2_ARTIFACTS=$W/art XE2_HOLD_POLL=2 PATH=$W/bin:$PATH; unset XE2_TOP XE2_PAIR_HELD
mkdir -p $HOME/.cache/gpu-lab $XE2_ARTIFACTS $W/bin; printf '#!/bin/bash\nexit 0\n' > $W/bin/vulkaninfo; chmod +x $W/bin/vulkaninfo
stage() { local D=$W/art/stage/$1; mkdir -p $D/{parent,cand}
  for b in parent cand; do printf '#!/bin/bash\necho STUB_RUNNER_LAUNCHED >> %s/launched\n' $W > $D/$b/llama_main; chmod +x $D/$b/llama_main; : > $D/$b/libllama_runner.so; : > $D/$b/env; done
  : > $D/prompt_2048.txt; : > $D/prompt_check.txt; echo test > $D/STAGE.md; echo $D; }
run() { bash $T/e2e5.sh --stage $1 --out raw --lock 00000000-test --clkmin 1 > $1/e2e5.out 2>&1 & echo $!; }
started() { [[ -s $1/raw/env.txt ]]; }
stop() { pkill -P $1 2>/dev/null; kill $1 2>/dev/null; wait $1 2>/dev/null; }

echo "== 1 coordinator hold (a HOLD file in the test's artifact directory)"
D=$(stage hold); touch $W/art/HOLD; p=$(run $D); sleep 8
ok "the session does not start under a hold" "! started $D"
ok "HELD names what would start next" "grep -q '^HELD 20.*e2e5.sh --stage' $W/art/HELD"
rm $W/art/HOLD; sleep 6
ok "it starts when the hold is removed" "started $D"
ok "and HELD is gone" "[[ ! -e $W/art/HELD ]]"; stop $p

echo "== 2 occupied pair lock (a cheap screen holds it shared, as on the other card)"
D=$(stage pair); ( . $T/host.sh; gpu_begin shared "screen"; sleep 10 ) & h=$!; sleep 1; p=$(run $D); sleep 5
ok "the session does not start while the pair lock is held" "! started $D && kill -0 $h"
wait $h; sleep 4
ok "it starts when the lock is released" "started $D"; stop $p

echo "== 3 a guard that cannot be acquired stops the runner"
D=$(stage noguard); mv $W/art/run $W/art/run.kept; : > $W/art/run          # run/ cannot be created: no pair lock
bash $T/e2e5.sh --stage $D --out raw --lock 00000000-test --clkmin 1 > $D/e2e5.out 2>&1; rc=$?
ok "status 75" "[[ $rc == 75 ]]"
ok "nothing was started" "! started $D && grep -q 'guard not acquired' $D/e2e5.out"
rm $W/art/run; mv $W/art/run.kept $W/art/run

echo "== 4 output"
ok "no 'command not found' in any runner output" "! grep -l 'command not found' $W/art/stage/*/e2e5.out"
ok "no stub runner was launched" "[[ ! -e $W/launched ]]"
ok "the campaign's real HOLD / HELD / pair lock were not touched" '[[ $before == "$(ls -la --time-style=full-iso $REAL/HOLD $REAL/HELD $REAL/run/pair.lock 2>&1)" ]]'
[[ $fails == 0 ]] && { echo "TEST_E2E5_GUARD_PASS"; rm -rf $W; exit 0; }; echo "TEST_E2E5_GUARD_FAIL ($fails), evidence in $W"; exit 1
