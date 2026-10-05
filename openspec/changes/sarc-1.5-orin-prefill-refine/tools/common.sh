#!/bin/bash
# common.sh: constants and probes of the Jetson Orin Nano campaign, sourced by every tool.
# Two sides share this file. The workstation (x86, build host) builds, stages, deploys, pulls and analyses; the
# device (duck-naughty, aarch64) only measures. The device holds a copy of this tools directory, the unmodified
# sarc/tools/verify.sh and the kit prompts under ~/hmz-sarc-orin/executorch/ at the same relative paths, so the
# measuring tools read the same $ET, $CHANGE, $KIT on both sides. Nothing is compiled on the device.
# Temperature comes from /sys/class/thermal (zone gpu-thermal), the clock from the devfreq node of the GPU, the
# load from the nvgpu platform node, power from the ina3221 rail VDD_IN. nvidia-smi reports N/A on a Jetson and
# `nvidia-smi pmon` hangs there: neither is used.
#
# Exit codes shared by the tools: 70 = the GPU sensors stopped answering (the run ends, nothing is retried; a
# marker file blocks every later tool until a person removes it), 75 = gpu-lab lock busy, 76 = a GPU process this
# campaign did not start, 77 = a required input is missing.
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEVICE=doremy@duck-naughty; DEVROOT=hmz-sarc-orin   # relative to the device's $HOME
CHANGE_REL=openspec/changes/sarc-1.5-orin-prefill-refine
if [[ $(uname -m) == aarch64 ]]; then SIDE=device; R=$HOME/$DEVROOT; A=$R
else SIDE=ws; R=/mnt/linux-share/hmz-campaigns/jetson; A=$R/.artifacts/orin-prefill-refine; fi
ET=$R/executorch; CHANGE=$ET/$CHANGE_REL
LOCK=b49259c9-868c-5b7c-b6f1-65a2bf4b63be
KIT=$ET/openspec/changes/sarc-1.5-e2e-benchmark/kit
PARENT_COMMIT=6a7cc8cc6
MODELDIR=$HOME/.cache/et-jetson-study/models   # device, flat layout, read-only
# Every process the campaign starts inherits this variable; others() uses it to tell our jobs from foreign ones.
export SARC_CAMPAIGN_TAG=orin-prefill-refine
export ETVK_DEVICE_INDEX=0
GONE=$A/GPU_GONE
[[ -f $GONE ]] && { echo "refusing to run: $(cat $GONE)" >&2; exit 70; }
need() { local f; for f in "$@"; do [[ -s $f ]] || { echo "missing required file: $f" >&2; exit 77; }; done; }
# dssh <command>: one command on the device (workstation side).
dssh() { ssh -o BatchMode=yes -o ConnectTimeout=20 $DEVICE "$@"; }
[[ $SIDE == ws ]] && return 0

# ---- device side only ----
GPUDEV=/sys/class/devfreq/17000000.gpu; GPULOAD=/sys/devices/platform/17000000.gpu/load
for GZ in /sys/class/thermal/thermal_zone*; do [[ $(cat $GZ/type 2>/dev/null) == gpu-thermal ]] && break; GZ=""; done
for HW in /sys/class/hwmon/hwmon*; do [[ $(cat $HW/name 2>/dev/null) == ina3221 ]] && break; HW=""; done
gtemp_m() { local t; read -r t < $GZ/temp 2>/dev/null || return 1; [[ $t =~ ^[0-9]+$ ]] || return 1; echo "$t"; }   # milli C
gtemp() { local t; t=$(gtemp_m) || return 1; echo $((t / 1000)); }
gclk() { local f; read -r f < $GPUDEV/cur_freq 2>/dev/null || return 1; echo $((f / 1000000)); }   # MHz
gpu_alive() { [[ -n $GZ ]] && gtemp > /dev/null && gclk > /dev/null; }
# gpu_gone <where>: record the loss (marker) and end the run. Never retry, reboot or reload drivers.
gpu_gone() {
  local msg="GPU_GONE $(date -u +%FT%TZ) $(hostname): GPU sensors failed ($1). Run ended; nothing retried. Recovery needs a person; remove $GONE afterwards."
  echo "$msg" >&2; mkdir -p $A; echo "$msg" >> $GONE; exit 70
}
gone_check() { [[ -f $GONE ]] && exit 70; gpu_alive || gpu_gone "${1:-health check}"; }
# sampler_start <file>: "epoch_us clock_MHz busy% power_W temp_C" every 0.1 s (runs take 2 to 63 s here); stop
# with sampler_stop. power = VDD_IN (whole module), temp = gpu-thermal.
sampler_start() {
  ( exec 9>&-; while :; do
      read -r f < $GPUDEV/cur_freq; read -r l < $GPULOAD; read -r t < $GZ/temp; read -r mv < $HW/in1_input; read -r ma < $HW/curr1_input
      mw=$((mv * ma / 1000)); printf '%s %d %d %d.%03d %d.%01d\n' "${EPOCHREALTIME/./}" $((f / 1000000)) $((l / 10)) $((mw / 1000)) $((mw % 1000)) $((t / 1000)) $((t % 1000 / 100))
      sleep 0.1
    done > "$1" 2>/dev/null ) & SP=$!
}
sampler_stop() { kill $SP 2>/dev/null; wait $SP 2>/dev/null; }
# mem_line: available memory and swap counters, recorded before and after every run (8 GB shared CPU/GPU).
mem_line() { awk '/^MemAvailable|^SwapFree/ {printf "%s=%d ", $1, $2 / 1024}' /proc/meminfo | tr -d ':'; awk '/^pswpin|^pswpout/ {printf "%s=%s ", $1, $2}' /proc/vmstat; }
mem_avail_mb() { awk '/^MemAvailable/ {print int($2 / 1024)}' /proc/meminfo; }

# own_pid <pid>: the process, or one of its ancestors, carries the campaign tag (a zombie of ours has no readable
# environment; a process that vanished before it could be identified is not reported). A process of another user
# has an unreadable environment and a foreign parent chain: it is reported.
own_pid() { local p=$1 st n=0
  while [[ $p -gt 1 && $n -lt 16 ]]; do
    st=$(sed 's/.*) //' /proc/$p/stat 2>/dev/null) || return 0
    { tr '\0' '\n' < /proc/$p/environ; } 2>/dev/null | grep -qx "SARC_CAMPAIGN_TAG=$SARC_CAMPAIGN_TAG" && return 0
    set -- $st; p=$2; n=$((n + 1))
  done; return 1; }
# others: known GPU programs by name, and a running GitHub Actions job (Runner.Worker; the idle Runner.Listener
# service is not a job), that this campaign did not start. There is no per-process GPU client list on a Jetson
# without root, so an unknown program using the GPU is not seen here; the GPU load before each run is recorded.
others() {
  local p
  for p in $( { pgrep -x 'llama_main|test_llama_micr|logits_dump|llama-server|ollama|vllm|roofline|roofline_sustai|inspect'
                pgrep -f 'ComfyUI|comfyui|Runner\.Worker'; } | sort -un ); do
    [[ -d /proc/$p ]] || continue
    own_pid $p || echo "$p:$(cat /proc/$p/comm 2>/dev/null):$(stat -c %U /proc/$p 2>/dev/null)"
  done | tr '\n' ';'
}
no_others() { local o; o=$(others); [[ -z $o ]] && return 0; abort_others "$1" "$o"; }
abort_others() { mkdir -p $A; echo "ABORT $(date -u +%FT%TZ) $1: GPU process not started by this campaign: $2" | tee -a $A/ABORTED >&2; exit 76; }
others_watch_start() { : > "$1"; ( while :; do o=$(others); [[ -n $o ]] && echo "$o" >> "$1"; sleep 0.5; done ) 9>&- & OW=$!; }
others_watch_stop() { kill $OW 2>/dev/null; wait $OW 2>/dev/null; others >> "$1"; tr ';' '\n' < "$1" | grep . | sort -u | tr '\n' ';'; }
# cool_to <target milli C> <timeout s>: wait until the GPU is at or below the target, the timeout has passed, or
# the temperature has stopped falling (less than 0.25 C in 20 s): the idle temperature of this fan-cooled module
# drifts, so a fixed target alone may never be reached.
cool_to() { local t0=$SECONDS t ref tref
  ref=$(gtemp_m) || gpu_gone cool_to; tref=$SECONDS
  while :; do
    t=$(gtemp_m) || gpu_gone cool_to
    (( t <= $1 || SECONDS - t0 >= $2 )) && break
    if (( SECONDS - tref >= 20 )); then (( ref - t < 250 )) && break; ref=$t; tref=$SECONDS; fi
    sleep 5
  done; }
# cool_start [timeout s]: before a job, wait until the temperature has stopped falling.
cool_start() { cool_to 0 ${1:-300}; }
take_lock() { exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK" || exit 75; flock -w ${1:-1800} 9 || { echo "gpu-lab lock busy" >&2; exit 75; }; }
