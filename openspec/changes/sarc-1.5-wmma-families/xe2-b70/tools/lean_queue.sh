#!/bin/bash
# lean_queue.sh <build tag>: the finite matrix under the owner decision of 2026-10-15 23:30 UTC, on one binary
# that holds the control and every profile. First the control: unmodified verify.sh --pdiff compared line by line
# with the historical final gate, and a seven-repeat A/A session that calibrates the clock floor. Then each
# profile of profiles.json (other member's picks first): verify.sh --pdiff (a production-diff failure rejects
# without timing), a seven-repeat interleaved session against the same-session control, the dispatched names,
# the evidence rows. A next-token difference alone is CORRECTNESS_PENDING and still timed. Resumable: a finished
# session has <session>/matrix.done; an unfinished one stops the queue for inspection.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"
TAG=${1:?build tag}
REPS=7
AA=s1-aa
INVENTORY=$XE2_ROOT/inputs/inventory-f393c3ff7f/manifests/xe2.json
HIST_GATE=${XE2_HIST_GATE:?historical final gate verify.out}
ORDER=(b70-wmma-dq-n13 b70-wmma-q4-table b70-wmma-dq-i b70-wmma-dq-table b70-wmma-q4-k16 b70-wmma-q4-k32 b70-wmma-dq-n26)
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$A/queue.log" > "$A/queue.status"; }
stop() { status "STOPPED $*"; exit 1; }
halted() { [[ -e $A/DEVICE_LOST ]] && stop "DEVICE_LOST: $(cat "$A/DEVICE_LOST")"
  [[ -e $A/FOREIGN_STOP ]] && stop "FOREIGN_STOP: $(head -1 "$A/FOREIGN_STOP")"; return 0; }
record() { python3 "$TOOLS/lean_record.py" "$A" "$C" "$AA" "${ORDER[@]}" > /dev/null
  git -C "$ET" add "$C/STATUS.md" "$C/results"
  git -C "$ET" commit -q -m "[SARC][docs] B70 lean matrix: $1" || true; }
[[ $(sha256sum "$INVENTORY" | cut -d' ' -f1) == 5904caa4c1977f256d58b4c1c111e7fdcb2e54bc312676b40c90da6bcf048306 ]] || stop 'manifest hash'
[[ ${#ORDER[@]} == $(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["profiles"]))' "$C/profiles.json") ]] || stop 'profile order list'
grep -qx BUILD_BOTH_OK "$A/build/$TAG.src.txt" || stop "build $TAG"
python3 "$TOOLS/input_provenance.py" "$INPUT_RECEIPT" "$XE2_MODEL_ROOT" || stop 'input record'
halted

S=$A/stage/$AA
if [[ ! -f $S/matrix.done ]]; then
  [[ ! -e $S ]] || stop "unfinished control session $AA: inspect before resuming"
  status "CONTROL stage+verify"
  "$TOOLS/stage.sh" $AA $TAG "$PARENT_ENV" $TAG "$PARENT_ENV" 'A/A: final profile in both arms on the matrix binary' > "$A/logs/$AA-stage.log" 2>&1 || stop 'stage control'
  "$TOOLS/matrix_verify.sh" $AA $TAG "$S/cand/env"; rc=$?; halted
  [[ $rc == 0 ]] || stop "control verify rc=$rc"
  python3 "$TOOLS/verify_diff.py" "$HIST_GATE" "$S/verify.out" > "$S/verify-vs-historical.txt" \
    || stop "control verify differs from the historical final gate: $S/verify-vs-historical.txt"
  status "CONTROL A/A session"
  "$TOOLS/session.sh" $AA --calibrate --reps $REPS > "$S/e2e5.out" 2>&1; rc=$?; halted
  [[ $rc == 0 ]] || stop "A/A session rc=$rc: $(tail -1 "$S/raw/done.txt" 2>/dev/null)"
  python3 "$TOOLS/matrix_evidence.py" "$A" $AA --candidate control > "$S/evidence.out" 2>&1 || stop "A/A evidence: $S/evidence.out"
  python3 -c 'import json, sys; sys.exit(not json.load(open(sys.argv[1]))["a_a_consistent"])' "$S/evidence.json" \
    || stop 'A/A outside +-2 % per cell or +-1 % geomean: explain before candidates'
  echo CONTROL_OK > "$S/correctness.txt"; echo COMPLETE > "$S/matrix.done"; record "control gate and A/A"
fi

for name in "${ORDER[@]}"; do
  S=$A/stage/$name
  [[ -f $S/matrix.done ]] && continue
  [[ ! -e $S ]] || stop "unfinished session $name: inspect before resuming"
  coordinator_hold "lean matrix $name"
  status "$name stage+verify"
  "$TOOLS/stage.sh" "$name" $TAG "$PARENT_ENV" $TAG "$PARENT_ENV ET_VK_SARC_B70_WMMA_PROFILE=$name" \
    'Finite Xe2 existing-kernel matrix; attention invariant' > "$A/logs/$name-stage.log" 2>&1 || stop "stage $name"
  "$TOOLS/matrix_verify.sh" "$name" $TAG "$S/cand/env"; rc=$?; halted
  python3 "$TOOLS/verify_diff.py" "$A/stage/$AA/verify.out" "$S/verify.out" > "$S/verify-vs-control.txt"
  if [[ $(grep -c '^pdiff .* rc=0 .*ALL PASSED' "$S/verify.out") != 12 ]]; then
    echo "REJECTED_PDIFF (verify rc=$rc)" > "$S/correctness.txt"; echo COMPLETE > "$S/matrix.done"
    record "$name rejected by production diff"; continue
  fi
  status "$name session"
  "$TOOLS/session.sh" "$name" --reps $REPS > "$S/e2e5.out" 2>&1; rc=$?; halted
  [[ $rc == 0 || $rc == 1 ]] || stop "$name session rc=$rc"
  status "$name dispatch"
  "$TOOLS/matrix_dispatch.sh" "$name" $TAG > "$S/dispatch.out" 2>&1; halted
  python3 "$TOOLS/matrix_evidence.py" "$A" "$name" --candidate "$name" --parent-verify "$A/stage/$AA/verify.out" \
    --inventory "$INVENTORY" > "$S/evidence.out" 2>&1
  if grep -q DIFFER "$S/raw/nexttoken.csv"; then label="CORRECTNESS_PENDING (next token DIFFER: $(grep DIFFER "$S/raw/nexttoken.csv" | cut -d, -f1,2 | tr '\n' ' '); top-2 logits not printed by llama_main)"
  elif [[ $rc == 0 ]]; then label="PASS_SCREEN (pdiff 12/12, next tokens SAME)"
  else label="SCREEN_INCOMPLETE"; fi
  [[ $rc == 0 ]] || label+="; session: $(tail -1 "$S/raw/done.txt")"
  echo "$label" > "$S/correctness.txt"; echo COMPLETE > "$S/matrix.done"; record "$name"
done
status 'MATRIX_COMPLETE'
