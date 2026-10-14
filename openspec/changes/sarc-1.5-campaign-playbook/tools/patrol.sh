#!/bin/bash
# patrol.sh: read-only patrol of the running GPU campaigns, for the coordinating session (COORDINATOR.md).
#
# For every campaign in the table below it prints: the branch head, the number of uncommitted files, the files
# changed outside the dev zone and under the gate / golden paths since the parent commit, the age and the top
# of STATUS.md, any "decision needed" / "blocking" section in it, the coordinator hold state (tools/HOLD.md),
# the GPU processes and temperature, markers that a queue left behind (ABORTED, GPU_GONE), and large files this
# user wrote under /tmp in the last day. Before that: the state of every hmz run on this machine and the local disk and load.
#
# It changes nothing on a host or in a working copy: every remote command is a read, sent as a script to
# `ssh -o BatchMode=yes <host> 'bash -s'` (never through the login shell, which may not be bash) under a timeout.
# The only thing it writes is one fingerprint file per campaign on THIS machine, to tell "no progress since
# the last patrol" (set PATROL_STATE= to switch that off).
#
# usage: patrol.sh [-n LINES]     LINES of STATUS.md to show per campaign (default 14)
#
# ---- configuration: edit this block, nothing below it -------------------------------------------------------
# One campaign per line: name  host  workdir  change-dir  [parent=<commit>] [artifacts=<dir>]
#   host        an ssh alias of the GPU host (key authentication, no prompt), or `local` for this machine
#   workdir     absolute path of the working copy on that host (the directory named `executorch`)
#   change-dir  the campaign's directory under openspec/changes/
#   parent      the parent commit of the campaign; without it the zone checks are skipped
#   artifacts   the artifact directory; default: `.artifacts` beside the working copy
CAMPAIGNS=(
  # "7900xtx  <host>  <abs path>/executorch  sarc-1.5-7900xtx-prefill-refine  parent=<commit>"
  # "rx7600   <host>  <abs path>/executorch  sarc-1.5-rx7600-prefill-refine   parent=<commit>"
)
HMZ_PY=${HMZ_PY-$HOME/.local/share/uv/tools/hmz/bin/python}   # empty: skip the hmz run states
PATROL_STATE=${PATROL_STATE-${XDG_STATE_HOME:-$HOME/.local/state}/campaign-patrol}
SSH_TIMEOUT=${SSH_TIMEOUT:-45}                               # seconds per host, connection included
GPU_PROGRAMS='llama_main|test_llama_micr|logits_probe|logits_dump|llama-bench|llama-completio|roofline'
# ---- end of configuration -----------------------------------------------------------------------------------
set -u
LINES_STATUS=14
while [[ $# -gt 0 ]]; do
  case $1 in
    -n) LINES_STATUS=${2:?-n needs a number}; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
[[ $LINES_STATUS =~ ^[0-9]+$ ]] || { echo "-n needs a number" >&2; exit 2; }
[[ ${#CAMPAIGNS[@]} -gt 0 ]] || { echo "no campaign configured: edit the CAMPAIGNS table at the top of $0" >&2; exit 2; }

echo "patrol $(date -u +%FT%TZ)"

# 1. hmz runs on this machine. A run that ended or failed is the first thing to look for: three model-call
#    failures in a row end a run, and a few minutes of provider trouble did that to two campaigns at once.
if [[ -n $HMZ_PY && -x $HMZ_PY ]]; then
  echo "== hmz runs"
  timeout 30 "$HMZ_PY" - <<'PY' 2>&1 | cut -c1-140
from hmz.sdk import Daemons
for d in Daemons().all():
    s = d.status()  # state names are hmz's own; an ended or failed run is flagged by substring
    u = s.get("usage") or {}
    state = str(s.get("state"))
    flag = "   <-- LOOK" if any(w in state.lower() for w in ("fail", "end", "stop", "error", "cancel")) else ""
    print(f"  {str(d.workspace).split('/')[-1]:<18} {state!s:<10} spent {str(u.get('duration'))[:9]:<9} out {u.get('output_tokens')}{flag}")
PY
else
  echo "== hmz runs: skipped (HMZ_PY is not set or not executable)"
fi
echo "== this machine: free $(df -h --output=avail . | tail -1 | tr -d ' ') here, $(uptime | sed 's/.*load/load/')"

# 2. One read-only script per campaign, run by bash on the host. Positional values: workdir change-dir parent
#    artifacts status-lines gpu-programs. Its last line is the fingerprint.
read -r -d '' REMOTE <<'EOF'
W=$1; C=$2; PARENT=$3; A=$4; N=$5; PROGS=$6
cd "$W" 2>/dev/null || { echo "  WORKING COPY MISSING: $W"; echo "FINGERPRINT missing"; exit 0; }
[ -n "$A" ] || A=$(dirname "$W")/.artifacts
head=$(git log -1 --format='%h %ad %s' --date=format:'%m-%d %H:%M' 2>/dev/null | cut -c1-150)
dirty=$(git status --short 2>/dev/null | wc -l)
echo "  branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null) | head $head"
echo "  uncommitted files: $dirty$([ "$dirty" -gt 25 ] && echo '   <-- piling up')"
if [ -n "$PARENT" ] && git cat-file -e "$PARENT^{commit}" 2>/dev/null; then
  out=$(git diff --name-only "$PARENT" HEAD | grep -v -E '^backends/vulkan/runtime/graph/ops/(glsl|impl)/sarc_dev/|^backends/vulkan/test/sarc_dev/|^sarc/|^openspec/')
  gate=$(git diff --name-only "$PARENT" HEAD | grep -E '^sarc/(tools|golden)/')
  echo "  outside the dev zone since $PARENT: $(echo "$out" | grep -c .)"; [ -n "$out" ] && echo "$out" | head -8 | sed 's/^/    /'
  [ -n "$gate" ] && { echo "  GATE OR GOLDEN FILES CHANGED   <-- LOOK"; echo "$gate" | head -8 | sed 's/^/    /'; }
  wgate=$(git status --short -- sarc/tools sarc/golden | head -4); [ -n "$wgate" ] && { echo "  uncommitted edits under sarc/tools or sarc/golden   <-- LOOK"; echo "$wgate" | sed 's/^/    /'; }
else
  echo "  zone checks skipped (no parent commit configured, or it is not in this clone)"
fi
S=openspec/changes/$C/STATUS.md
if [ -f "$S" ]; then
  age=$(( ($(date +%s) - $(stat -c %Y "$S")) / 60 ))
  echo "  STATUS.md: written $(date -u -r "$S" +%m-%dT%H:%MZ), ${age} min ago$([ "$age" -gt 240 ] && echo '   <-- stale?')"
  sed -n "1,${N}p" "$S" | grep -v '^$' | cut -c1-400 | sed 's/^/    /'
  grep -n -i -A5 -E '^#+ .*(decision needed|blocking|waiting for the owner)' "$S" | head -14 | cut -c1-300 | sed 's/^/    ! /'
else
  echo "  STATUS.md: not written yet"
fi
[ -e "$A/HOLD" ] && echo "  HOLD present since $(date -u -r "$A/HOLD" +%m-%dT%H:%MZ)$([ -e "$A/HELD" ] && echo "; queue reports: $(head -1 "$A/HELD" | cut -c1-160)" || echo '; queue has NOT acknowledged yet (a job is still running, or the queue has no hold)')"
[ ! -e "$A/HOLD" ] && [ -e "$A/HELD" ] && echo "  HELD without HOLD: the queue should have removed it   <-- LOOK"
for m in ABORTED GPU_GONE; do [ -s "$A/$m" ] && echo "  marker $m: $(tail -1 "$A/$m" | cut -c1-200)   <-- LOOK"; done
procs=$(pgrep -a -x "$PROGS" 2>/dev/null | cut -c1-110 | head -4)
echo "  GPU programs running: $(echo "$procs" | grep -c .)"; [ -n "$procs" ] && echo "$procs" | sed 's/^/    /'
if command -v nvidia-smi >/dev/null 2>&1; then
  g=$(timeout 10 nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu --format=csv,noheader 2>&1 | head -2 | tr '\n' ';')
  echo "  gpu: $g"; case $g in *[Ee]rror*|*[Ff]ailed*|*"Unable"*|"") echo "  DEVICE NOT ANSWERING   <-- LOOK: do not retry, reboot or reload drivers";; esac
else
  n=0
  for h in /sys/class/hwmon/hwmon*; do
    case $(cat "$h/name" 2>/dev/null) in amdgpu|xe|i915)
      n=$((n + 1)); t=$(cat "$h/temp1_input" 2>/dev/null || cat "$h/temp2_input" 2>/dev/null)
      echo "  gpu sensor $(cat "$h/name"): ${t:+$((t / 1000)) C} busy $(cat "$h/device/gpu_busy_percent" 2>/dev/null || echo '?') %";;
    esac
  done
  [ "$n" = 0 ] && echo "  gpu: no hwmon sensor of a known GPU driver found (device gone, or a sensor this script does not know)"
fi
gov=$(for f in /sys/class/drm/card*/device/power_dpm_force_performance_level /sys/class/devfreq/*gpu*/governor; do [ -r "$f" ] && echo "$(basename "$f")=$(cat "$f")"; done 2>/dev/null | sort -u | tr '\n' ' ')
[ -n "$gov" ] && echo "  clock policy: $gov"
tmpbig=$(find /tmp -xdev -user "$(id -un)" -type f -size +20M -mmin -1440 2>/dev/null | head -5)
[ -n "$tmpbig" ] && { echo "  files over 20 MB written under /tmp by this user in the last day (artifacts belong in $A)   <-- LOOK"; echo "$tmpbig" | sed 's/^/    /'; }
if [ -d "$A" ]; then echo "  artifact directory: $(df -h --output=avail "$A" 2>/dev/null | tail -1 | tr -d ' ') free"; else echo "  artifact directory missing: $A"; fi
newest=$(find "$A" -maxdepth 3 -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
echo "FINGERPRINT $(git rev-parse --short HEAD 2>/dev/null) dirty=$dirty status=$(stat -c %Y "$S" 2>/dev/null) newest_artifact=${newest:-none}"
EOF

[[ -n $PATROL_STATE ]] && mkdir -p "$PATROL_STATE" 2>/dev/null
for line in "${CAMPAIGNS[@]}"; do
  read -r -a f <<<"$line"
  [[ ${#f[@]} -ge 4 ]] || { echo "== bad table line (need name host workdir change-dir): $line"; continue; }
  name=${f[0]}; host=${f[1]}; work=${f[2]}; change=${f[3]}
  parent=""; artifacts=""
  for kv in "${f[@]:4}"; do
    case $kv in parent=*) parent=${kv#parent=} ;; artifacts=*) artifacts=${kv#artifacts=} ;; *) echo "== $name: unknown field $kv" ;; esac
  done
  echo "== $name  [$host]"
  # The arguments travel inside the script, so nothing but `bash -s` passes through the host's login shell.
  script="set -- $(printf '%q ' "$work" "$change" "$parent" "$artifacts" "$LINES_STATUS" "$GPU_PROGRAMS")"$'\n'"$REMOTE"
  if [[ $host == local ]]; then
    out=$(timeout "$SSH_TIMEOUT" bash -s <<<"$script" 2>&1); rc=$?
  else
    out=$(timeout "$SSH_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" 'bash -s' <<<"$script" 2>&1); rc=$?
  fi
  if [[ $rc != 0 ]]; then
    echo "  HOST NOT REACHED (rc $rc$([[ $rc == 124 ]] && echo ', timeout'))   <-- LOOK: $(echo "$out" | tail -1 | cut -c1-160)"
    continue
  fi
  echo "$out" | grep -v '^FINGERPRINT '
  fp=$(echo "$out" | grep '^FINGERPRINT ' | tail -1)
  if [[ -n $PATROL_STATE && -n $fp ]]; then
    f=$PATROL_STATE/$name.fingerprint
    if [[ -s $f && $(sed -n 1p "$f") == "$fp" ]]; then
      echo "  NO CHANGE since the patrol of $(sed -n 2p "$f") (same head, same STATUS.md, no newer artifact)   <-- LOOK if a job should be running"
    else
      printf '%s\n%s\n' "$fp" "$(date -u +%FT%TZ)" > "$f"
    fi
  fi
done
