#!/bin/bash
# Existing exports only: parent snapshot, topic gate and five-repeat six-cell A/A.
set -euo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"
PB=wmma-b70-parent-01
CB=wmma-b70-topic-02
NAME=s1-aa-01
[[ ! -e $A/DEVICE_LOST && ! -e $A/FOREIGN_STOP && ! -e $A/QUEUE_BLOCKED ]]
[[ ! -e $A/stage/s0-parent-verify && ! -e $A/stage/$NAME ]]
mkdir -p "$A/preflight" "$A/logs"
for item in $PARENT_ENV; do echo "$item"; done > "$A/preflight/control.env"
{
  echo "orchestration_commit=$(git -C "$ET" rev-parse HEAD)"
  sha256sum "$TOOLS"/*.{sh,py} "$A/build/$CB/identity/wmma_linear_identity"
  cat "$A/build/$PB.src.txt" "$A/build/$CB.src.txt"
} > "$A/preflight/baseline-queue-provenance.txt"
coordinator_hold 'parent snapshot'
"$TOOLS/matrix_verify.sh" s0-parent-verify "$PB" "$A/preflight/control.env"
if rg -i 'VK_ERROR_(DEVICE_LOST|OUT_OF_DEVICE_MEMORY)|ErrorDeviceLost|ErrorOutOfDeviceMemory' "$A/stage/s0-parent-verify/verify"; then
  echo 'Parent snapshot contains a device/allocation failure; inspect before any further GPU launch' >&2
  exit 1
fi
coordinator_hold 'A/A topic gate'
"$TOOLS/stage.sh" "$NAME" "$PB" "$PARENT_ENV" "$CB" "$PARENT_ENV" \
  'Parent versus unchanged topic control; final linear and attention profile in both arms' > "$A/logs/$NAME-stage.log"
"$TOOLS/matrix_verify.sh" "$NAME" "$CB" "$A/stage/$NAME/cand/env"
python3 "$TOOLS/verify_diff.py" "$A/stage/s0-parent-verify/verify.out" "$A/stage/$NAME/verify.out" \
  > "$A/stage/$NAME/verify-diff.txt"
coordinator_hold 'A/A calibration'
"$TOOLS/session.sh" "$NAME" --calibrate --reps 5 > "$A/stage/$NAME/e2e5.out" 2>&1
python3 "$TOOLS/matrix_evidence.py" "$A" "$NAME" --candidate control \
  --parent-verify "$A/stage/s0-parent-verify/verify.out" > "$A/stage/$NAME/evidence.out" 2>&1
echo 'BASELINE_EVIDENCE_RECORDED; actor must assess baseline differences, A/A consistency and repeat rule before candidates'
