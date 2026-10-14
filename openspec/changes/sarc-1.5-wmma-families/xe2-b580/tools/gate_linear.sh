#!/bin/bash
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"
if [[ ${WMMA_UNIT:-} != "$0" ]]; then
  exec "$TOOLS/hold.sh" run "gate_linear.sh ${1:-}" env WMMA_UNIT="$0" "$0" "$@"
fi

S=${1:?session}; D=$A/stage/$S
device_check || exit 77
idle_wait "gate $S" || exit $?
gpu_shared || exit 75
[[ -s $D/llama_main.bin && -s $D/test_llama_microbench && -s $D/env ]] || exit 2
[[ ! -e $D/verify.out ]] || { echo "gate output already exists; preserve it and use a new session"; exit 2; }
cool_start
python3 "$TOOLS/sampler.py" "$D/verify.sensors" 0.01 "$B580_TOP" & sampler_pid=$!
trap 'kill "$sampler_pid" 2>/dev/null; wait "$sampler_pid" 2>/dev/null' EXIT
guarded "$D/verify.others" python3 "$TOOLS/clean_exec.py" "$D/env" "$ET/sarc/tools/verify.sh" \
  --dir "$D" --lock "$LOCK" --models 1b,3b,8b --schemes 4w,8da4w --pdiff --out verify > "$D/verify.out" 2>&1
rc=$?
echo "VERIFY_DONE rc=$rc" >> "$D/verify.out"
kill "$sampler_pid" 2>/dev/null; wait "$sampler_pid" 2>/dev/null
trap - EXIT
device_check || exit 77
[[ $rc == 0 ]] || { echo "GATE_STOP rc=$rc" > "$D/gate.done"; exit "$rc"; }
python3 - "$D" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
rows = [line.split() for line in (p / "verify.sensors").read_text().splitlines() if line.startswith("D ")]
assert len(rows) >= 2, "missing foreign engine samples"
assert int(rows[-1][3]) == 0, "foreign engine activity during gate"
assert not (p / "verify.others").read_text().strip(), "foreign compute activity"
assert (p / "verify/done.txt").exists(), "incomplete verify gate"
print("VERIFY_RECORDED; compare with parent snapshot before acceptance")
PY
rc=$?
echo "GATE_RECORDED rc=$rc" > "$D/gate.done"
exit "$rc"
