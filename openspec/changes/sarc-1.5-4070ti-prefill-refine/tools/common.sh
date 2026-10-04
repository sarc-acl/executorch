#!/bin/bash
# common.sh: host constants and probes of the RTX 4070 Ti SUPER campaign (gpu-dev-4004), sourced by every tool.
# Temperature, clock and power come from nvidia-smi (no hwmon on this driver).
#
# Exit codes shared by the tools: 70 = the card stopped answering nvidia-smi (Xid 79 rule: the run ends, nothing
# is retried; a marker file blocks every later tool until a person removes it), 75 = gpu-lab lock busy,
# 76 = a GPU process this campaign did not start, 77 = a required input is missing.
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=$HOME/hmz-sarc-4070ti
A=$R/.artifacts/4070ti-prefill-refine; ET=$R/executorch
CHANGE=$ET/openspec/changes/sarc-1.5-4070ti-prefill-refine
LOCK=81a511a2-de7e-c3c8-f641-3562c315ffa7
KIT=$ET/openspec/changes/sarc-1.5-e2e-benchmark/kit
PARENT_COMMIT=6a7cc8cc6
# Every process the campaign starts inherits this variable; others() uses it to tell our jobs from foreign ones.
export SARC_CAMPAIGN_TAG=4070ti-prefill-refine
export SARC_MOUNT_ROOT=$R ETVK_DEVICE_INDEX=0
GONE=$A/GPU_GONE
[[ -f $GONE ]] && { echo "refusing to run: $(cat $GONE)" >&2; exit 70; }

gtemp() { local t; t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  [[ $t =~ ^[0-9]+$ ]] || return 1; echo "$t"; }
gpu_alive() { gtemp > /dev/null; }
# gpu_gone <where>: record the loss (marker, STATUS.md) and end the run. Never retry, reboot or reload drivers.
gpu_gone() {
  local msg="GPU_GONE $(date -u +%FT%TZ) $(hostname): nvidia-smi failed ($1). Run ended; nothing retried. Recovery needs a person; remove $GONE afterwards."
  echo "$msg" >&2; mkdir -p $A; echo "$msg" >> $GONE
  printf '\n## DEVICE LOST\n\n%s\n' "$msg" >> $CHANGE/STATUS.md
  exit 70
}
# gone_check: call after every child; a child that hit gpu_gone ends the caller too.
gone_check() { [[ -f $GONE ]] && exit 70; gpu_alive || gpu_gone "${1:-health check}"; }

own_pid() { tr '\0' '\n' < /proc/$1/environ 2>/dev/null | grep -qx "SARC_CAMPAIGN_TAG=$SARC_CAMPAIGN_TAG"; }
# others: GPU clients (compute apps and graphics/Vulkan clients from pmon) and known GPU programs by name that
# were NOT started by this campaign (no campaign tag in their environment, or environment unreadable).
others() {
  local p
  for p in $( { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null
                nvidia-smi pmon -c 1 2>/dev/null | awk '$1 != "#" && $2 ~ /^[0-9]+$/ {print $2}'
                pgrep -x 'llama_main|test_llama_micr|llama-server|ollama|vllm'
                pgrep -f 'ComfyUI|comfyui|Runner\.Worker'; } | sort -un ); do
    [[ -d /proc/$p ]] || continue
    own_pid $p || echo "$p:$(cat /proc/$p/comm 2>/dev/null)"
  done | tr '\n' ';'
}
# no_others <where>: stop measuring when a foreign GPU process is present (reported, never forced).
no_others() { local o; o=$(others); [[ -z $o ]] && return 0; abort_others "$1" "$o"; }
# abort_others <where> <captured>: end the run on an observation already made; it is not repeated, because the
# process may be gone by now and the interference still happened.
abort_others() { mkdir -p $A; echo "ABORT $(date -u +%FT%TZ) $1: GPU process not started by this campaign: $2" | tee -a $A/ABORTED >&2; exit 76; }
# others_watch_start <file> / others_watch_stop <file>: look for foreign GPU processes every 0.5 s while a job
# runs (a probe before and after cannot show that none was there in between). stop takes one last look and
# prints every distinct pid:name seen, empty when none. A process that lives less than the polling interval
# between two looks can still be missed.
others_watch_start() { : > "$1"; ( while :; do o=$(others); [[ -n $o ]] && echo "$o" >> "$1"; sleep 0.5; done ) 9>&- & OW=$!; }
others_watch_stop() { kill $OW 2>/dev/null; wait $OW 2>/dev/null; others >> "$1"; tr ';' '\n' < "$1" | grep . | sort -u | tr '\n' ';'; }
# cool_start [max C] [timeout s]: wait until the GPU is at or below the given temperature.
cool_start() { local t0=$SECONDS t
  while :; do t=$(gtemp) || gpu_gone cool_start; (( t > ${1:-50} && SECONDS - t0 < ${2:-300} )) || break; sleep 5; done; }
take_lock() { exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK" || exit 75; flock -w ${1:-1800} 9 || { echo "gpu-lab lock busy" >&2; exit 75; }; }
need() { local f; for f in "$@"; do [[ -s $f ]] || { echo "missing required file: $f" >&2; exit 77; }; done; }
