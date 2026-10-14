# host.sh adapter for the RTX 4070 Ti SUPER, for kit/session.sh. Lock, sensors (nvidia-smi: clocks.gr,
# utilization, power, temperature every 20 ms) and the Xid 79 rule are the tuning campaign's: if the card stops
# answering nvidia-smi the session ends and nothing is retried. The campaign's clock floors are per cell, from
# its calibration; this adapter uses their minimum, read from the campaign's clkmin file when it is there.
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../common-simple.sh"
LOCK=81a511a2-de7e-c3c8-f641-3562c315ffa7
CLKJSON=${CAMPAIGN_CLKMIN:-$HOME/hmz-sarc-4070ti/executorch/openspec/changes/sarc-1.5-4070ti-prefill-refine/results/4070ti/clkmin.json}
DEV_CLKMIN=$(python3 -c 'import json,sys; print(min(c["clkmin_mhz"] for c in json.load(open(sys.argv[1]))["cells"].values()))' "$CLKJSON" 2>/dev/null)
DEV_CLKMIN=${DEV_CLKMIN:-2400}; DEV_BUSYMAX=""
# A 1B prefill lasts 60 to 70 ms on this card and nvidia-smi answers every 20 ms at best: the campaign accepts a
# window with a few samples, and so does this adapter (two), where the other devices ask for five.
export MIN_CLK_SAMPLES=2
export ETVK_DEVICE_INDEX=0
gtemp() { local t; t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null) || { echo "GPU not answering nvidia-smi: stop, do not retry" >&2; kill -TERM $TOP; exit 70; }; echo "$t"; }
# epoch_us clock_Hz busy power_uW temp_mC from nvidia-smi's 20 ms loop. The pipeline runs as children of this
# function's subshell and is killed with it: without the trap, killing the sampler left nvidia-smi polling on.
dev_sampler() { trap 'pkill -P $BASHPID 2>/dev/null; exit 0' TERM
  stdbuf -oL nvidia-smi --query-gpu=clocks.gr,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits -lms 20 2>/dev/null \
  | while IFS=', ' read -r c u p t; do printf '%s %d %s %d %d\n' "${EPOCHREALTIME/./}" "$((c * 1000000))" "$u" "$(printf '%.0f' "$(echo "$p * 1000000" | bc 2>/dev/null || echo 0)")" "$((t * 1000))"; done > "$1" 2>/dev/null &
  wait; }
