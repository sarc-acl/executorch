#!/bin/bash
# test_gate_finite.sh <stage/<session> with a passing --sdpa gate>: regression test of gate_check.py's
# finite-error requirement. Copies the session's small gate files (no binaries), checks that the copy passes,
# then that the gate fails when one [sdpa-error] record of one pass reads nan, reads inf, or is missing, while
# its [sdpa-correctness] line still says mismatches=0 PASSED (what the test prints for a NaN output).
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"
S=$(readlink -f "$1"); N=$(basename $S); W=$A/tmp/test_gate_finite.$$; P=$A/stage/s0-parent-verify/verify.out; fails=0
fresh() { rm -rf $W; mkdir -p $W/$N; (cd $S && find . -maxdepth 1 -type f -size -200k ! -name 'lib*' -exec cp {} $W/$N/ \; ; cp -r verify raw trace sdpa-correctness cand cand-traced $W/$N/ 2>/dev/null
  rm -rf $W/$N/raw/logs $W/$N/trace/*/*.etdp $W/$N/cand/llama_main $W/$N/cand/*.so 2>/dev/null); }
run() { python3 $TOOLS/gate_check.py $W/$N --sdpa --parent $P > $W/out.txt 2>&1; echo $?; }
expect() { # <label> <want rc> <want pattern>
  local rc; rc=$(run); if [[ $rc == $2 ]] && grep -q "$3" $W/out.txt; then echo "ok   $1"; else echo "FAIL $1 (rc $rc)"; tail -3 $W/out.txt; fails=$((fails+1)); fi; }
fresh; expect "unmodified copy passes" 0 '^GATE_PASS'
L=sdpa-correctness/cand-full-r7.log
for v in nan -nan inf; do fresh; sed -i "0,/^\[sdpa-error\]/s/rms_err=[^ ]*/rms_err=$v/" $W/$N/$L
  expect "rms_err=$v in one record fails" 1 '^FAIL sdpa: tier full.*r7: .*error records'; done
fresh; sed -i "0,/^\[sdpa-error\]/s/ref_rms=[^ ]*/ref_rms=nan/" $W/$N/$L; expect "ref_rms=nan in one record fails" 1 '^FAIL sdpa: tier full.*r7: .*error records'
fresh; sed -i "0,/^\[sdpa-error\]/{/^\[sdpa-error\]/d}" $W/$N/$L; expect "a missing record fails" 1 '^FAIL sdpa: tier full.*r7: 3 error records'
fresh; grep -c 'mismatches=0/' $W/$N/$L | grep -qx 4 && echo "ok   the edited log still reads mismatches=0 in its 4 cases" || { echo "FAIL mismatch lines"; fails=$((fails+1)); }
rm -rf $W
[[ $fails == 0 ]] && echo "TEST_GATE_FINITE_OK" || { echo "TEST_GATE_FINITE_FAIL ($fails)"; exit 1; }
