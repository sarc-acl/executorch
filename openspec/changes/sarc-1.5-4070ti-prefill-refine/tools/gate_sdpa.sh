#!/bin/bash
# gate_sdpa.sh <session> ["<cand env>"]: gate.sh for a candidate that changes an SDPA kernel. The same steps,
# preceded by what sarc/tools/verify.sh does not run:
#   0a. test_llama_microbench --sdpa-correctness-only, tiers extended and full, 12 passes each with the candidate
#       env; accepted only with the full case count per pass, 0 mismatches and pairing=ok on every case
#       (gate_check.py sdpa);
#   0b. the SDPA perf suite (--sdpa) with and without the candidate env, for the dispatched kernels and times.
source "$(dirname "$0")/common.sh"; source $TOOLS/gatelib.sh; S=$1; D=$A/stage/$S; B=$D/test_llama_microbench
[[ -e $D/gate.done ]] && { echo "session $S already gated: $(cat $D/gate.done)" >&2; exit 2; }
need $D/STAGE.md $B $PARENT_CTL/verify.out $CLKFILE; cand_env "${@:2}"; grep -q GATE_ACCEPTED $PARENT_CTL/gate.done || { echo "no accepted parent control" >&2; exit 77; }
O=$D/sdpa-correctness; mkdir -p $O
sdpa_pass() { cool_start 60 300; env $ENVS $TOOLS/gl.sh $B --sdpa-correctness-only --sdpa-tier=$1 > $O/cand-$1-r$2.log 2>&1; }
for tier in extended full; do for i in $(seq 1 12); do
  step "sdpa $tier r$i" sdpa_pass $tier $i; echo "cand $tier r$i rc=0" >> $O/summary.txt
done; done
step sdpa-check python3 $TOOLS/gate_check.py sdpa $O $D/cand/env > $D/sdpa-check.txt 2>&1
perf() { cool_start 60 300; env $1 $TOOLS/gl.sh $B --sdpa --json-out=$O/perf-$2.json > $O/perf-$2.log 2>&1; }
for x in "cand:$ENVS" "table:"; do   # recorded only; device loss, a busy lock or a foreign process still end the gate
  perf "${x#*:}" ${x%%:*}; rc=$?; echo "perf ${x%%:*} rc=$rc" >> $O/summary.txt
  [[ $rc == 70 || $rc == 75 || $rc == 76 || -f $GONE ]] && finish GATE_ABORTED "sdpa perf ${x%%:*} rc=$rc" $rc
done
cool_start 50 300
step verify run_verify "$ENVS"
step verify-check python3 $TOOLS/gate_check.py verify $D/verify.out $PARENT_CTL/verify.out > $D/verify-check.txt 2>&1
timed_and_traced
finish GATE_ACCEPTED "all steps passed; the gain is in raw/summary.csv" 0
