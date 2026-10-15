#!/bin/bash
# matrix_queue.sh <build tag>: finite reviewed profiles, after the actor records baseline readiness.
set -euo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"
TAG=${1:?build tag}
[[ -s $A/BASELINE_READY && -s $A/reps && -s $CLKMIN_FILE && ! -e $A/DEVICE_LOST && ! -e $A/QUEUE_BLOCKED && ! -e $A/RESOURCE_LIMIT_BLOCKED ]]
[[ -x $A/build/$TAG/identity/wmma_linear_identity ]]
grep -qx BUILD_BOTH_OK "$A/build/$TAG.src.txt"
PV=$A/stage/s0-parent-verify/verify.out
[[ -s $PV ]]
INVENTORY=${XE2_INVENTORY:?reviewed manifest}
[[ $(sha256sum "$INVENTORY" | cut -d' ' -f1) == 5904caa4c1977f256d58b4c1c111e7fdcb2e54bc312676b40c90da6bcf048306 ]]
status() { echo "$(date -u +%FT%TZ) $*" | tee "$A/queue.status"; }
block() {
  [[ ! -e $A/QUEUE_BLOCKED ]] || return 0
  echo "$*" > "$A/QUEUE_BLOCKED"
  status "BLOCKED $*"
  python3 - "$C/STATUS.md" "$*" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
s += '\n## Decision needed from the owner\n\nFinite matrix stopped: ' + sys.argv[2] + '.\nInspect the recorded failure and decide whether to authorize a resume. Raw evidence is preserved; no automatic restart.\n'
p.write_text(s)
PY
}
trap 'rc=$?; if [[ $rc != 0 ]]; then block "job failed with rc=$rc; inspect artifact queue log before any further launch"; fi' EXIT
while IFS='|' read -r name scheme; do
  [[ ! -e $A/DEVICE_LOST ]]
  coordinator_hold "finite matrix $name"
  S=$A/stage/$name
  if [[ -f $S/matrix.done ]]; then
    python3 - "$S" "$(cat "$A/reps")" <<'PY_CHECK'
import collections, csv, json, pathlib, sys
s = pathlib.Path(sys.argv[1]); repeats = int(sys.argv[2])
assert (s/'raw/done.txt').read_text().splitlines()[-1].strip() == 'E2E5_OK'
text = (s/'raw/runs.csv').read_text(); assert text.endswith('\n')
rows = list(csv.DictReader(text.splitlines()))
assert all(None not in r and all(v is not None for v in r.values()) for r in rows)
counts = collections.Counter((r['model'],r['scheme'],r['build']) for r in rows if r['log'].startswith('logs/prefill') and r['valid']=='1')
assert all(counts[(m,q,b)]>=repeats for m in ('1b','3b','8b') for q in ('4w','8da4w') for b in ('parent','cand'))
assert not json.loads((s/'evidence.json').read_text())['issues']
assert (s/'verify.out').read_text().rstrip().endswith('VERIFY_DONE rc=0')
assert (s/'identity.log').read_text().rstrip().endswith('LINEAR_IDENTITY_PASS')
assert len(list(csv.reader((s/'identity.csv').open()))) == 24
for m in ('1b','3b','8b'):
    for q in ('4w','8da4w'):
        for b in ('parent','cand'):
            assert (s/f'dispatch/{m}-{q}-{b}.json').is_file()
assert (s/'dispatch/dispatch.done').stat().st_size
PY_CHECK
    status "SKIP_COMPLETE $name"
    continue
  fi
  [[ ! -e $S ]] || { block "incomplete existing session $name needs owner-approved resume"; exit 1; }
  status "STAGING $name"
  CE="$PARENT_ENV ET_VK_SARC_B70_WMMA_PROFILE=$name"
  "$TOOLS/stage.sh" "$name" "$TAG" "$PARENT_ENV" "$TAG" "$CE" 'Finite Xe2 existing-kernel matrix; attention invariant' > "$A/logs/$name-stage.log" 2>&1
  status "VERIFY $name"
  "$TOOLS/matrix_verify.sh" "$name" "$TAG" "$S/cand/env"
  status "IDENTITY $name"
  env $PARENT_ENV "$TOOLS/gl.sh" "$A/build/$TAG/identity/wmma_linear_identity" "$name" "$scheme" > "$S/identity.log" 2>&1
  grep -q '^LINEAR_IDENTITY_PASS$' "$S/identity.log"
  grep '^llama-' "$S/identity.log" > "$S/identity.csv"
  status "TIMING $name"
  "$TOOLS/session.sh" "$name" --reps "$(cat "$A/reps")" > "$S/e2e5.out" 2>&1
  status "DISPATCH $name"
  "$TOOLS/matrix_dispatch.sh" "$name" "$TAG" > "$S/dispatch.out" 2>&1
  status "RECORD $name"
  python3 "$TOOLS/matrix_evidence.py" "$A" "$name" --candidate "$name" --parent-verify "$PV" --inventory "$INVENTORY" > "$S/evidence.out" 2>&1
  echo COMPLETE > "$S/matrix.done"
  python3 - "$C/STATUS.md" "$S/evidence.json" <<'PY'
import datetime, json, pathlib, sys
path = pathlib.Path(sys.argv[1]); result = json.loads(pathlib.Path(sys.argv[2]).read_text())
s = path.read_text()
s += '\n' + datetime.datetime.now(datetime.timezone.utc).isoformat() + ': completed ' + result['candidate'] + '; independent measured review pending.\n\n'
s += '| Cell | Control tok/s | Candidate tok/s | Slowdown |\n|---|---:|---:|---:|\n'
for row in result['cells']:
    s += f"| {row['model']} {row['scheme']} | {row['control_tok_s']:.2f} | {row['candidate_tok_s']:.2f} | {row['slowdown']:.2%} |\n"
s += f"\nSix-cell geometric-mean slowdown {result['geomean_slowdown']:.2%}; point estimates within member limits: {result['point_estimates_within_member_margin']}. No family-wide acceptance.\n"
path.write_text(s)
PY
  git -C "$ET" add "$C/STATUS.md"
  git -C "$ET" commit -m "[SARC][docs] B70 finite matrix result $name"
  status "COMPLETE $name"
done < <(python3 - "$C/profiles.json" <<'PY'
import json, sys
for profile in json.load(open(sys.argv[1]))['profiles']:
    print(profile['name'] + '|' + ('8da4w' if '-dq-' in profile['name'] else '4w'))
PY
)
status 'MATRIX_COMPLETE; fresh independent measured review required'
