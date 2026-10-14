#!/bin/bash
# common.sh: host constants and probes of the RTX 4070 Ti SUPER fused-attention port campaign, sourced by every tool.
# Temperature, clock and power come from nvidia-smi (no hwmon on this driver).
#
# Exit codes shared by the tools: 70 = the card stopped answering nvidia-smi (Xid 79 rule: the run ends, nothing
# is retried; a marker file blocks every later tool until a person removes it), 75 = gpu-lab lock busy,
# 76 = a GPU process this campaign did not start, 77 = a required input is missing.
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=$HOME/hmz-sarc-4070ti-fused
A=$R/.artifacts; ET=$R/executorch
CHANGE=$ET/openspec/changes/sarc-1.5-4070ti-fused-port
LOCK=81a511a2-de7e-c3c8-f641-3562c315ffa7
KIT=$ET/openspec/changes/sarc-1.5-e2e-benchmark/kit
# The parent of every comparison is this commit WITH this environment (the first campaign's accepted stack);
# the pristine arm of the closing session is the same build with no environment.
PARENT_COMMIT=6050b1287
PARENT_ENV="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=4070ti-refine1"
export TMPDIR=$A/tmp; mkdir -p $TMPDIR
# Every process the campaign starts inherits this variable; others() uses it to tell our jobs from foreign ones.
export SARC_CAMPAIGN_TAG=4070ti-fused-port
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

# own_pid <pid>: the process, or one of its ancestors, carries the campaign tag. Ancestors are needed twice:
# a process that has exited but is not yet reaped (state Z) has no readable environment (a runner of ours caught
# between exit and wait was reported as foreign in the first parent control, superseded/zombie-sighting), and a
# tool we start may launch its children with a cleaned environment (igpu-roofline's `roofline` and `inspect`
# runners, reported during the first roofline run). A process under a foreign parent is still foreign.
own_pid() { local p=$1 st n=0
  while [[ $p -gt 1 && $n -lt 16 ]]; do
    # A process that is gone before it could be identified cannot be attributed to anyone: it is not reported
    # (the third false abort, s2-c1: a pid with no name left, seen while verify.sh was starting and ending
    # runners of ours). This is the documented limit: a process shorter than one look can be missed.
    st=$(sed 's/.*) //' /proc/$p/stat 2>/dev/null) || return 0
    { tr '\0' '\n' < /proc/$p/environ; } 2>/dev/null | grep -qx "SARC_CAMPAIGN_TAG=$SARC_CAMPAIGN_TAG" && return 0
    set -- $st; p=$2; n=$((n + 1))
  done; return 1; }
# others: GPU clients (compute apps and graphics/Vulkan clients from pmon) and known GPU programs by name that
# were NOT started by this campaign (no campaign tag in their environment, or environment unreadable).
others() {
  local p
  for p in $( { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null
                nvidia-smi pmon -c 1 2>/dev/null | awk '$1 != "#" && $2 ~ /^[0-9]+$/ {print $2}'
                pgrep -x 'llama_main|test_llama_micr|llama-server|ollama|vllm'
                pgrep -f 'ComfyUI|comfyui|Runner\.Worker'; } | sort -un ); do
    [[ -d /proc/$p ]] || continue
    own_pid $p || echo "$p:$(cat /proc/$p/comm 2>/dev/null):$(stat -c %U /proc/$p 2>/dev/null)"
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
# Owner decision 2026-10-06 (00:35 UTC): the runner's abort after its output happens only in processes that have to
# read their model from the share (STATUS.md, "a slow model load"), so the measuring tools may read the model into
# the page cache first, for both arms alike. cached_pct <file>: share of the file in the page cache.
# warm_model <pte> [<csv> [<tag>]]: when the model changes (the first process of a cell), and whenever the file is
# not fully cached, map it and touch every page (warm_file.py: a plain read does not hold on this host, see there),
# at most 3 passes; one row utc,file,cached_before_pct,passes,cached_after_pct,tag is appended to <csv> whether or
# not anything had to be read.
cached_pct() { fincore -n -b -o RES,SIZE "$1" 2>/dev/null | awk '{printf "%d", 100 * $1 / $2}'; }
warm_model() { local f=$1 b a n=0; b=$(cached_pct "$f"); a=$b
  [[ ${WARM_LAST:-} == "$f" ]] || { python3 $TOOLS/warm_file.py "$f"; n=1; a=$(cached_pct "$f"); WARM_LAST=$f; }
  while [[ ${a:-0} -lt 100 && $n -lt 3 ]]; do python3 $TOOLS/warm_file.py "$f"; n=$((n + 1)); a=$(cached_pct "$f"); done
  [[ -z ${2:-} ]] || echo "$(date -u +%FT%TZ),$(basename "$f"),$b,$n,$a,${3:-}" >> "$2"; }
# hold_point <what starts next>: the coordinator hold (owner decision 2026-10-06, playbook tools/HOLD.md). While
# $A/HOLD exists nothing new starts: HELD is written, the device lock (fd 9, when this shell holds it) is released
# for the coordinator's measurement and taken again afterwards. Called before every unit: a build, a verify.sh,
# an SDPA pass, a cell of a timed session, a traced run, a screen run. SARC_HOLD_NAME=HOLD-TEST tests the path.
hold_point() { local n=${SARC_HOLD_NAME:-HOLD} w=0 l=0; local h=$A/$n d=$A/HELD${n#HOLD}
  while [[ -e $h ]]; do
    if [[ $w == 0 ]]; then { : >&9; } 2>/dev/null && { flock -u 9; l=1; }; echo "HELD $(date -u +%FT%TZ) $1" > $d; w=1; fi
    sleep ${SARC_HOLD_POLL:-60}
  done
  [[ $w == 1 ]] && { rm -f $d; [[ $l == 1 ]] && { flock -w 7200 9 || { echo "gpu-lab lock busy after hold" >&2; exit 75; }; }; }
  return 0; }
take_lock() { exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK" || exit 75; flock -w ${1:-1800} 9 || { echo "gpu-lab lock busy" >&2; exit 75; }; }
need() { local f; for f in "$@"; do [[ -s $f ]] || { echo "missing required file: $f" >&2; exit 77; }; done; }
