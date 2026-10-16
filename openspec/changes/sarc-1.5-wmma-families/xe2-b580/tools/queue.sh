#!/bin/bash
# queue.sh: detached, resumable finite matrix of the lean owner decision (2026-10-15 23:30 UTC).
# Build wmma-v1 once (control and every candidate profile in one binary), then:
#   s1-aa: pristine parent-v2 vs wmma-v1, both with the control profile; unmodified verify on wmma-v1
#          (control check) and the calibrating A/A session.
#   s<N>-<cand>: wmma-v1 control vs wmma-v1 candidate; unmodified verify first; a production-diff failure
#          rejects the candidate untimed; otherwise the six-cell interleaved session (7 valid runs per arm).
# Every finished unit is skipped on restart. State: $A/queue/wmma-v1/job-state.json.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"
Q=$A/queue/wmma-v1; mkdir -p "$Q"; STATE=$Q/job-state.json
BUILD=wmma-v1; REPS=7; EXTRA=5
CANDS=(b70-linear dq-n26 q4-narrow q4-band dq-table dq-i)
state() { python3 - "$STATE" "$@" <<'PY'
import json, sys, datetime
path, status, current = sys.argv[1:4]
try: d = json.load(open(path))
except Exception: d = {"started": datetime.datetime.utcnow().isoformat() + "Z", "units": []}
d.update(status=status, current=current, pid=int(sys.argv[4]), updated=datetime.datetime.utcnow().isoformat() + "Z")
if len(sys.argv) > 5: d["units"].append(sys.argv[5])
json.dump(d, open(path + ".tmp", "w"), indent=1); __import__("os").replace(path + ".tmp", path)
PY
}
DONE=0
trap '[[ $DONE == 1 ]] || state failed "${CUR:-?}" $$ "queue exited during ${CUR:-?}"' EXIT
step() { CUR=$1; state running "$1" $$; }
note() { state running "$CUR" $$ "$(date -u +%FT%TZ) $*"; echo "$(date -u +%FT%TZ) $*"; }

step "build $BUILD"
if ! grep -qx BUILD_BOTH_OK "$A/build/$BUILD.src.txt" 2>/dev/null; then
  [[ -e $A/build/$BUILD.src.txt ]] && { note "build $BUILD exists but failed; new tag needed"; exit 1; }
  "$TOOLS/build-both.sh" $BUILD "$(git -C "$ET" rev-parse HEAD)" || { note "build $BUILD failed"; exit 1; }
fi
note "build $BUILD ok"

stage() { [[ -s $A/stage/$1/STAGE.md ]] || "$TOOLS/stage.sh" "$@" > /dev/null; }
lost() { [[ -e $A/DEVICE_LOST ]] && { note "DEVICE_LOST: $(cat "$A/DEVICE_LOST")"; exit 1; }; return 0; }
gate() { [[ -s $A/stage/$1/verify.out ]] || "$TOOLS/gate_linear.sh" "$1"; note "gate $1 rc=$? $(cat "$A/stage/$1/gate.done" 2>/dev/null)"; lost; }
session() { local s=$1 c=$2; shift 2
  [[ -s $A/stage/$s/raw/done.txt ]] || B580_CANDIDATE=$c "$TOOLS/session.sh" "$s" --reps $REPS --extra $EXTRA "$@"
  note "session $s: $(tail -1 "$A/stage/$s/raw/done.txt" 2>/dev/null)"; lost; }

step "s1-aa"
stage s1-aa parent-v2 "$PARENT_ENV" $BUILD "$PARENT_ENV" "A/A: pristine base vs wmma-v1, control profile both arms" || { note "stage s1-aa failed"; exit 1; }
gate s1-aa
# s1-aa's session (2026-10-16 00:05-00:41Z) is preserved: every prefill invalid (foreign_busy, Firefox).
# s1b-aa (foreign-quiet wait, no rows) is under stage/superseded/. A/A under the coordinator's BUSYMAX ruling:
stage s1c-aa parent-v2 "$PARENT_ENV" $BUILD "$PARENT_ENV" "A/A calibrating BUSYMAX (min 5 %, 2 x p90); control check is s1-aa's gate on the same binary" || { note "stage s1c-aa failed"; exit 1; }
session s1c-aa aa --calibrate
[[ -s $CLKMIN_FILE && -s $BUSYMAX_FILE ]] || { note "calibration missing; candidates not started"; exit 1; }

i=2
for c in "${CANDS[@]}"; do
  s=s$i-$c; i=$((i + 1)); step "$s"
  stage $s $BUILD "$PARENT_ENV" $BUILD "ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-fused1-order-wmma-$c" "candidate $c vs control, one binary" \
    || { note "stage $s failed"; continue; }
  gate $s
  v=$A/stage/$s/verify.out
  if ! grep -q '^VERIFY_DONE rc=0' "$v" 2>/dev/null; then
    echo "GATE_INCOMPLETE $(tail -1 "$v" 2>/dev/null)" > "$A/stage/$s/screen.txt"; note "$s gate incomplete; not timed"; continue
  fi
  if [[ $(grep -c '^pdiff .* rc=0 \[production-diff\] ALL PASSED' "$v" 2>/dev/null) != 12 ]]; then
    echo "REJECTED production-diff $(grep '^pdiff' "$v" 2>/dev/null | grep -vc 'ALL PASSED') failing lines" > "$A/stage/$s/screen.txt"
    note "$s rejected: production diff"; continue
  fi
  session $s $c
done
DONE=1; state completed "done" $$ "$(date -u +%FT%TZ) queue completed"
