#!/bin/bash
# matrix.sh <tag> <build tag> <candidate ids...>: the finite 780M matrix as one resumable detached chain
# (owner decision 2026-10-15 23:30 UTC). Units, each under the coordinator hold and the gpu-lab lock, each
# skipped when its marker exists:
#   1. ctrl-verify: unmodified sarc/tools/verify.sh --pdiff on the control (780m-final)
#   2. aa: control vs control session (6 cells, 7 valid runs per arm, no clock floor); then the floor =
#      97 % of the lowest per-run median clock of the valid A/A runs
#   3. per candidate: verify.sh --pdiff with the candidate environment; a production-diff failure rejects the
#      candidate (not timed); otherwise a control vs candidate session with the floor; summary, record, commit
# State: <artifacts>/jobs/<tag>/state.json (queued/running/completed/failed).
set -uo pipefail
TAG=$1; BUILD=$2; shift 2; CANDS=("$@")
T=$(cd "$(dirname "$0")" && pwd); C=$(dirname "$T"); ET=$(git -C "$T" rev-parse --show-toplevel)
ART=$(dirname "$ET")/.artifacts; A=$ART/matrix/$TAG; J=$ART/jobs/$TAG; B=$ART/build/$BUILD
HIST=$HOME/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-08/stage/r3b-final-dev15-head4/verify.out  # final gate, read only
LOCK=00000000-c400-0000-0000-000000000000; CTRL="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=780m-final"
mkdir -p "$A/stage" "$J"; echo "${CANDS[*]}" > "$A/ORDER"
for v in $(env | grep -oE '^(ET_VK|ETVK)[A-Z0-9_]*'); do unset "$v"; done
export TMPDIR=$ART/tmp
state() {  # state <status> <unit>
  python3 - "$J/state.json" "$1" "$2" "$TAG" <<'PY'
import datetime, json, os, sys
p, status, unit, tag = sys.argv[1:5]
s = json.load(open(p)) if os.path.exists(p) else {"tag": tag, "done": []}
s.update(status=status, unit=unit, pid=os.getppid(), updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         monitoring_visibility="own_user_plus_device_busy_other_users_unobservable",
         monitoring_owner_decision="2026-10-15 03:57 UTC")
if status == "running" and unit.startswith("done:"): s["done"].append(unit[5:])
open(p + ".tmp", "w").write(json.dumps(s, indent=1) + "\n"); os.replace(p + ".tmp", p)
PY
}
trap 'state failed "exit at ${UNIT:-start}"' EXIT
stage() {  # stage <session> "<cand env>"
  local S=$A/stage/$1 b
  mkdir -p "$S"/{parent,cand}
  for b in parent cand; do
    cp -f "$B/llama/examples/models/llama/llama_main" "$B/llama/examples/models/llama/runner/libllama_runner.so" "$S/$b/"
    cat "$B/COMMIT" > "$S/$b/COMMIT"
  done
  tr ' ' '\n' <<< "$CTRL" > "$S/parent/env"; tr ' ' '\n' <<< "$2" > "$S/cand/env"
  cp -f "$S/cand/llama_main" "$S/cand/libllama_runner.so" "$B/tests/test_llama_microbench" "$S/"
  cp -f "$ET"/openspec/changes/sarc-1.5-e2e-benchmark/kit/prompts/prompt_*.txt \
    "$ET"/openspec/changes/sarc-1.5-4070ti-prefill-refine/tools/r1304.txt "$S/"
  { echo "session $1 staged $(date -u +%FT%TZ) build $BUILD commit $(cat "$B/COMMIT")"
    echo "parent (control) env [$CTRL]"; echo "cand env [$2]"
    sha256sum "$S"/{parent,cand}/llama_main "$S"/{parent,cand}/libllama_runner.so "$S/test_llama_microbench"; } > "$S/STAGE.md"
}
verify() {  # verify <session> "<env>"
  local S=$A/stage/$1
  [[ -f $S/verify.done ]] && return 0
  rm -rf "$S/verify"
  env $2 bash "$T/hold.sh" run "verify.sh $1" bash "$ET/sarc/tools/verify.sh" --dir "$S" --lock $LOCK \
    --models 1b,3b,8b --schemes 4w,8da4w --pdiff --out verify > "$S/verify.out" 2>&1
  echo "VERIFY_DONE rc=$?" >> "$S/verify.out"; touch "$S/verify.done"
}
session() {  # session <session> <clkmin> <candidate label>
  local S=$A/stage/$1 t0=$SECONDS
  [[ -f $S/session.done ]] && return 0
  [[ -d $S/raw ]] && mv "$S/raw" "$S/raw.interrupted-$(date -u +%Y%m%dT%H%M%S)"
  while (( $(cat /sys/class/hwmon/hwmon2/temp1_input) > 48000 && SECONDS - t0 < 300 )); do sleep 5; done
  bash "$T/session.sh" --stage "$S" --out raw --lock $LOCK --reps 7 --extra 3 --clkmin "$2" --session "$1" \
    --candidate "$3" > "$S/e2e5.out" 2>&1
  grep -q E2E5_DONE "$S/e2e5.out" && touch "$S/session.done"
}
record() {
  python3 "$T/record.py" "$A" "$C"
  git -C "$ET" add -- "$C/STATUS.md" "$C/results/matrix" && \
    git -C "$ET" commit -q -m "[SARC][wmma] 780M matrix: record $1" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" \
    -- "$C/STATUS.md" "$C/results/matrix" || true
}

state running start
UNIT=ctrl-verify; state running $UNIT
stage ctrl-verify "$CTRL"; verify ctrl-verify "$CTRL"
python3 - "$A/stage/ctrl-verify/verify.out" "$HIST" > "$A/stage/ctrl-verify/compare-historical.txt" <<'PY'
import re, sys
strip = lambda p: [re.sub(r"(tok_s=)[0-9.]+", r"\1*", l.rstrip("\n")) for l in open(p)]
a, b = strip(sys.argv[1]), strip(sys.argv[2])
d = [f"now:  {x}" for x in a if x not in b] + [f"hist: {x}" for x in b if x not in a]
print("IDENTICAL (rates removed)" if not d else "DIFFERENT (rates removed)"); print("\n".join(d))
PY
state running done:ctrl-verify
UNIT=aa; state running $UNIT
[[ -d $A/stage/aa ]] || stage aa "$CTRL"
session aa 0 control-aa
[[ -f $A/stage/aa/session.done ]] || { UNIT="aa session incomplete"; exit 1; }
python3 "$T/summarize.py" "$A/stage/aa/raw" - - "$A/stage/aa/summary.json" > "$A/stage/aa/summary.txt" 2>&1
[[ -f $A/floor.txt ]] || python3 - "$A/stage/aa/raw/runs.csv" > "$A/floor.txt" <<'PY'
import csv, sys
c = [float(r["clk_med_mhz"]) for r in csv.DictReader(open(sys.argv[1])) if r["log"].startswith("logs/prefill") and r["valid"] == "1"]
print(int(0.97 * min(c)))
PY
FLOOR=$(cat "$A/floor.txt"); [[ $FLOOR -gt 0 ]] || { UNIT="no clock floor"; exit 1; }
record aa; state running done:aa
for c in "${CANDS[@]}"; do
  S=$A/stage/$c; UNIT=$c; state running "$c"
  [[ -f $S/candidate.done ]] && continue
  n=${c#rdna3-}; E="$CTRL ET_VK_SARC_780M_WMMA_PROFILE=780m-wmma-rdna3-$n"
  [[ -f $S/STAGE.md ]] || stage "$c" "$E"
  verify "$c" "$E"
  if ! grep '^pdiff' "$S/verify.out" | awk 'END {exit NR != 12} !/ rc=0 / || !/ALL PASSED/ {exit 1}'; then
    echo "REJECTED (production diff failed; not timed)" > "$S/verdict.txt"
  else
    session "$c" "$FLOOR" "$c"
    python3 "$T/summarize.py" "$S/raw" "$S/verify.out" "$A/stage/ctrl-verify/verify.out" "$S/summary.json" > "$S/summary.txt" 2>&1
    python3 - "$S/summary.json" > "$S/verdict.txt" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
v = "pdiff PASS"
v += "; next token SAME" if s["next_token_all_same"] else "; CORRECTNESS_PENDING (next token differs; top-2 logits not available from llama_main)"
if not s["complete"]: v += "; INCOMPLETE cells"
print(v)
PY
  fi
  touch "$S/candidate.done"; record "$c"; state running "done:$c"
done
trap - EXIT; UNIT=end; state completed end
