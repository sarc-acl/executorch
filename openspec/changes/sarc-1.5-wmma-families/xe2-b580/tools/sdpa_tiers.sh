#!/bin/bash
# sdpa_tiers.sh <session>: the SDPA part of a finalist's full gate (coordinator decision 2026-10-16 05:20 UTC):
# 12 passes each of test_llama_microbench --sdpa-correctness-only with tiers all, extended and full under the
# staged candidate environment, as sarc-1.5-b580-fused-port/tools/sdpa_passes.sh; decided by sdpa_check.py.
# Logs: stage/<session>/sdpa-correctness/cand-<tier>-r<i>.log, rc.csv; verdict: sdpa.done.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/host.sh"
if [[ ${WMMA_UNIT:-} != "$0" ]]; then
  exec "$TOOLS/hold.sh" run "sdpa_tiers.sh ${1:-}" env WMMA_UNIT="$0" "$0" "$@"
fi

S=${1:?session}; D=$A/stage/$S; O=$D/sdpa-correctness
device_check || exit 77
gpu_shared || exit 75
[[ -s $D/test_llama_microbench && -s $D/cand/env ]] || exit 2
[[ ! -e $O ]] || { echo "sdpa output already exists; preserve it and use a new session"; exit 2; }
mkdir -p "$O"
exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK"; flock -w 1800 9 || { echo "gpu-lab lock busy"; exit 75; }
for tier in all extended full; do for i in $(seq 1 12); do
  cool_start
  guarded "$O/others-$tier-r$i.txt" python3 "$TOOLS/clean_exec.py" "$D/cand/env" "$D/test_llama_microbench" \
    --sdpa-correctness-only --sdpa-tier=$tier > "$O/cand-$tier-r$i.log" 2>&1
  rc=$?; echo "cand,$tier,$i,$rc" >> "$O/rc.csv"; echo "sdpa cand $tier r$i rc=$rc"
  device_check || exit 77
  [[ $rc == 76 ]] && { echo "SDPA_STOP foreign GPU workload" > "$D/sdpa.done"; exit 76; }
done; done
python3 "$TOOLS/sdpa_check.py" "$O" > "$D/sdpa.check.txt"; rc=$?
echo "SDPA_RECORDED rc=$rc $(tail -1 "$D/sdpa.check.txt")" > "$D/sdpa.done"
exit $rc
