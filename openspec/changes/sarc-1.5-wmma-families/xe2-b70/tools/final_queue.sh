#!/bin/bash
# final_queue.sh <build tag>: the Xe2 family finalists of the coordinator decision of 2026-10-16 05:20 UTC. Builds
# the tag from HEAD once if it does not exist, then per finalist: unmodified verify.sh --pdiff, 12 passes of each
# SDPA correctness tier (all, extended, full) with the candidate environment, a seven-repeat interleaved session
# against the matrix control (next tokens on the three prompts), dispatched names, evidence rows, and
# gate_check.py against the control's verify output. Resumable: a finished session has <session>/matrix.done.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"
TAG=${1:?build tag}
REPS=7
AA=s1-aa
INVENTORY=$XE2_ROOT/inputs/inventory-f393c3ff7f/manifests/xe2.json
CV=$A/stage/$AA/verify.out
MATRIX=(b70-wmma-dq-n13 b70-wmma-q4-table b70-wmma-dq-i b70-wmma-dq-table b70-wmma-q4-k16 b70-wmma-q4-k32 b70-wmma-dq-n26)
FINAL=(xe2-family-a xe2-family-b)
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$A/queue.log" > "$A/queue.status"; }
stop() { status "STOPPED $*"; exit 1; }
halted() { [[ -e $A/DEVICE_LOST ]] && stop "DEVICE_LOST: $(cat "$A/DEVICE_LOST")"
  [[ -e $A/FOREIGN_STOP ]] && stop "FOREIGN_STOP: $(head -1 "$A/FOREIGN_STOP")"; return 0; }
record() { python3 "$TOOLS/lean_record.py" "$A" "$C" "$AA" "${MATRIX[@]}" "${FINAL[@]}" > /dev/null
  git -C "$ET" add "$C/STATUS.md" "$C/results"
  git -C "$ET" commit -q -m "[SARC][docs] B70 Xe2 finalists: $1" || true; }
[[ $(sha256sum "$INVENTORY" | cut -d' ' -f1) == 5904caa4c1977f256d58b4c1c111e7fdcb2e54bc312676b40c90da6bcf048306 ]] || stop 'manifest hash'
[[ -f $A/stage/$AA/matrix.done && -s $CLKMIN_FILE ]] || stop 'matrix control and calibration missing'
python3 "$TOOLS/input_provenance.py" "$INPUT_RECEIPT" "$XE2_MODEL_ROOT" || stop 'input record'
halted
if [[ ! -e $A/build/$TAG.src.txt ]]; then
  status "BUILD $TAG"
  "$TOOLS/build-both.sh" "$TAG" HEAD > "$A/logs/build-$TAG.log" 2>&1 || stop "build $TAG failed: $A/logs/build-$TAG.log"
fi
grep -qx BUILD_BOTH_OK "$A/build/$TAG.src.txt" || stop "build $TAG"

for name in "${FINAL[@]}"; do
  S=$A/stage/$name
  [[ -f $S/matrix.done ]] && continue
  [[ ! -e $S ]] || stop "unfinished session $name: inspect before resuming"
  coordinator_hold "finalist $name"
  CE="$PARENT_ENV ET_VK_SARC_B70_WMMA_PROFILE=$name"
  status "$name stage+verify"
  "$TOOLS/stage.sh" "$name" "$TAG" "$PARENT_ENV" "$TAG" "$CE" \
    'Xe2 family finalist (coordinator 2026-10-16 05:20 UTC); attention invariant' > "$A/logs/$name-stage.log" 2>&1 || stop "stage $name"
  "$TOOLS/matrix_verify.sh" "$name" "$TAG" "$S/cand/env"; halted
  python3 "$TOOLS/verify_diff.py" "$CV" "$S/verify.out" > "$S/verify-vs-control.txt"
  status "$name sdpa tiers"
  "$TOOLS/sdpa_passes.sh" "$name" cand 12 "$CE" > "$S/sdpa.out" 2>&1; rc=$?; halted
  [[ $rc == 75 || $rc == 76 ]] && stop "$name sdpa passes rc=$rc"
  status "$name session"
  "$TOOLS/session.sh" "$name" --reps $REPS > "$S/e2e5.out" 2>&1; rc=$?; halted
  [[ $rc == 0 || $rc == 1 ]] || stop "$name session rc=$rc"
  status "$name dispatch"
  "$TOOLS/matrix_dispatch.sh" "$name" "$TAG" > "$S/dispatch.out" 2>&1; halted
  python3 "$TOOLS/matrix_evidence.py" "$A" "$name" --candidate "$name" --parent-verify "$CV" \
    --inventory "$INVENTORY" > "$S/evidence.out" 2>&1
  python3 "$TOOLS/gate_check.py" "$S" --sdpa --parent "$CV" > "$S/gate.txt" 2>&1
  label="$(tail -1 "$S/gate.txt"); $(grep -c '^FAIL' "$S/gate.txt") FAIL line(s), see gate.txt"
  grep -q DIFFER "$S/raw/nexttoken.csv" && label+="; next token DIFFER: $(grep DIFFER "$S/raw/nexttoken.csv" | cut -d, -f1,2 | tr '\n' ' ')"
  echo "$label" > "$S/correctness.txt"; echo COMPLETE > "$S/matrix.done"; record "$name"
done
status 'FINALISTS_COMPLETE'
