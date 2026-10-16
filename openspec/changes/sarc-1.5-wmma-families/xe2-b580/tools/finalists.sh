#!/bin/bash
# finalists.sh: detached, resumable full gate and confirming session of the Xe2 family finalists (coordinator
# decision 2026-10-16 05:20 UTC). Build wmma-v2 once (control, every matrix candidate and both finalists), then
# per finalist s<N>-<finalist>: wmma-v2 control vs wmma-v2 finalist; unmodified verify.sh --pdiff (a
# production-diff failure stops the finalist), the SDPA tiers (sdpa_tiers.sh), then the six-cell interleaved
# session (7 valid runs per arm, next token on three prompts). Cells short of valid runs are re-timed in
# s<N>b-<finalist>. Every finished unit is skipped on restart. State: $A/queue/wmma-v2/job-state.json.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"
Q=$A/queue/wmma-v2; mkdir -p "$Q"; STATE=$Q/job-state.json
BUILD=wmma-v2; REPS=7; EXTRA=5
FINALISTS=(xe2-family-a xe2-family-b)
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
[[ -s $CLKMIN_FILE && -s $BUSYMAX_FILE && -s $IDLE_FILE ]] || { CUR=calibration; note "s1c-aa calibration missing"; exit 1; }

step "build $BUILD"
if ! grep -qx BUILD_BOTH_OK "$A/build/$BUILD.src.txt" 2>/dev/null; then
  [[ -e $A/build/$BUILD.src.txt ]] && { note "build $BUILD exists but failed; new tag needed"; exit 1; }
  "$TOOLS/build-both.sh" $BUILD "${FINALIST_COMMIT:?}" || { note "build $BUILD failed"; exit 1; }
fi
note "build $BUILD ok"

lost() { [[ -e $A/DEVICE_LOST ]] && { note "DEVICE_LOST: $(cat "$A/DEVICE_LOST")"; exit 1; }; return 0; }
i=8
for f in "${FINALISTS[@]}"; do
  s=s$i-$f; i=$((i + 1)); step "$s"; CE="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-fused1-order-wmma-$f"
  [[ -s $A/stage/$s/STAGE.md ]] || "$TOOLS/stage.sh" $s $BUILD "$PARENT_ENV" $BUILD "$CE" "finalist $f vs control, one binary" > /dev/null \
    || { note "stage $s failed"; continue; }
  [[ -s $A/stage/$s/verify.out ]] || "$TOOLS/gate_linear.sh" $s; note "gate $s rc=$? $(cat "$A/stage/$s/gate.done" 2>/dev/null)"; lost
  v=$A/stage/$s/verify.out
  grep -q '^VERIFY_DONE rc=0' "$v" 2>/dev/null || { note "$s verify incomplete; stopped"; continue; }
  if [[ $(grep -c '^pdiff .* rc=0 \[production-diff\] ALL PASSED' "$v") != 12 ]]; then
    echo "REJECTED production-diff" > "$A/stage/$s/screen.txt"; note "$s rejected: production diff"; continue
  fi
  [[ -s $A/stage/$s/sdpa.done ]] || "$TOOLS/sdpa_tiers.sh" $s; note "sdpa $s rc=$? $(cat "$A/stage/$s/sdpa.done" 2>/dev/null)"; lost
  [[ -s $A/stage/$s/raw/done.txt ]] || B580_CANDIDATE=$f "$TOOLS/session.sh" $s --reps $REPS --extra $EXTRA
  note "session $s: $(tail -1 "$A/stage/$s/raw/done.txt" 2>/dev/null)"; lost
  r=s$((i - 1))b-$f; d=$(tail -1 "$A/stage/$s/raw/done.txt" 2>/dev/null)
  [[ $d == E2E5_INCOMPLETE* ]] || continue
  cells=(${d#E2E5_INCOMPLETE}); printf '%s\n' "${cells[@]}" | grep -qv ':incomplete$' && { note "$s: not repairable ($d)"; continue; }
  step "$r"
  [[ -s $A/stage/$r/STAGE.md ]] || "$TOOLS/stage.sh" $r $BUILD "$PARENT_ENV" $BUILD "$CE" "re-time of $s cells: ${cells[*]}" > /dev/null \
    || { note "stage $r failed"; continue; }
  for cell in "${cells[@]}"; do cell=${cell%:incomplete}
    [[ -s $A/stage/$r/raw-$cell/done.txt ]] || B580_CANDIDATE=$f "$TOOLS/session.sh" $r --out raw-$cell --models ${cell%-*} --schemes ${cell#*-} --reps $REPS --extra $EXTRA
    note "session $r $cell: $(tail -1 "$A/stage/$r/raw-$cell/done.txt" 2>/dev/null)"; lost
  done
done
DONE=1; state completed "done" $$ "$(date -u +%FT%TZ) queue completed"
