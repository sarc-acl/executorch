# host.sh: host constants and guards of the Xe2 campaign (fedora-gpu-eval, card b70-0), sourced by every tool here.
# b70-0 is guest PCI 0000:01:00.0 = Vulkan device 0 (deviceUUID 868023e2-0000-0000-0100-000000000000).
TOOLS=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")"); C=$(dirname $TOOLS)
ET=$(cd $C/../../.. && pwd); XE2_ROOT=$(dirname $ET)          # ~/hmz-sarc-xe2/executorch, ~/hmz-sarc-xe2
A=${XE2_ARTIFACTS:-$XE2_ROOT/.artifacts}
LOCK=868023e2-0000-0000-0100-000000000000
PARENT_COMMIT=6a7cc8cc6
PDEV=0000:01:00.0; PCI=/sys/bus/pci/devices/$PDEV
HW=$(echo $PCI/hwmon/hwmon*); FREQ=$PCI/tile0/gt0/freq0
# Calibration written by `e2e5.sh --calibrate` (the baseline session) and required by every later session:
IDLE_FILE=$A/idle_temp_mc     # package temperature of the cool, idle card (millidegrees C)
CLKMIN_FILE=$A/clkmin_mhz     # lowest accepted median GT clock of a timed run (MHz)
export ETVK_DEVICE_INDEX=0 SARC_MOUNT_ROOT=$XE2_ROOT XE2_TOP=${XE2_TOP:-$$}
gtemp_mc() { cat $HW/temp2_input; }   # package temperature, millidegrees C
# cool_start: wait (at most 5 min) until the package is within 3 C of the calibrated idle temperature; before
# the calibration exists, until the temperature has not moved by more than 1 C over 60 s.
cool_start() { local t0=$SECONDS a b
  if [[ -s $IDLE_FILE ]]; then
    while (( $(gtemp_mc) > $(<$IDLE_FILE) + 3000 && SECONDS - t0 < 300 )); do sleep 5; done
  else
    while (( SECONDS - t0 < 300 )); do a=$(gtemp_mc); sleep 60; b=$(gtemp_mc); (( a - b <= 1000 && b - a <= 1000 )) && break; done
  fi; }
# export_commit <git work tree> <commit> <dest>: write the files of exactly that commit into <dest>, then every
# submodule at the commit the tree pins, recursively. Everything comes from the git object stores (read-tree +
# checkout-index into a scratch index), so modified, untracked or differently checked-out files in the live
# trees cannot reach the export. Appends "<path> <commit>" per tree to <dest>.export-manifest. Fails if a
# pinned submodule commit is not available locally.
export_commit() { local wt=$1 sha=$2 dest=$3 rel=${4:-.} idx p s
  git -C "$wt" cat-file -e "$sha^{commit}" 2>/dev/null || { echo "export: $rel: commit $sha not available in $wt" >&2; return 1; }
  idx=$(mktemp) && rm -f $idx && mkdir -p "$dest" || return 1
  GIT_INDEX_FILE=$idx git -C "$wt" read-tree "$sha" && GIT_INDEX_FILE=$idx git -C "$wt" checkout-index -a -f --prefix="$dest/" \
    || { rm -f $idx; echo "export: $rel: checkout of $sha failed" >&2; return 1; }
  rm -f $idx; echo "$rel $sha" >> "${XE2_EXPORT_MANIFEST:?}"
  while read -r s p; do
    [[ -e $wt/$p/.git ]] || { echo "export: submodule $rel/$p is not initialised" >&2; return 1; }
    mkdir -p "$dest/$p" && export_commit "$wt/$p" "$s" "$dest/$p" "$rel/$p" || return 1
  done < <(git -C "$wt" ls-tree -r "$sha" | awk '$2 == "commit" {print $3, $4}'); }
# gpu_others: GPU workloads that this campaign job did not start, as "pid:command;" entries; empty = the card
# is ours. A process counts when
#   - it holds a DRM file of b70-0 (any /proc/<pid>/fdinfo entry with drm-pdev = PDEV), whatever its name; or
#   - its command line names a known GPU workload. This second rule is the fallback for processes of other
#     users, whose fdinfo an unprivileged user cannot read (the fleet's LLM services run as this user).
# Ours = the top-level tool (XE2_TOP), its descendants, and its ancestors (the shells that launched it).
# Not counted: an idle monitor (monitor_idle). The owner's nvtop was open on this host before the campaign
# started; it holds a DRM file of both cards to read their counters and submits nothing. It is exempt only
# while every DRM client it owns shows zero engine cycles and zero GPU memory; the moment either is non-zero it
# is a foreign GPU process like any other. Exempt monitors are listed by gpu_monitors and recorded per session.
monitor_idle() { [[ $(ps -o comm= -p $1 2>/dev/null) == nvtop ]] || return 1
  ! awk '/^drm-(cycles|total|resident|shared|active)-[a-z0-9]+:/ && $2 + 0 > 0 {f = 1} END {exit !f}' /proc/$1/fdinfo/* 2>/dev/null; }
gpu_monitors() { local p; for p in $(grep -l -s "^drm-pdev:[[:space:]]*$PDEV" /proc/[0-9]*/fdinfo/* | cut -d/ -f3 | sort -un); do
  monitor_idle $p && printf '%s:%s;' $p "$(ps -o comm= -p $p 2>/dev/null)"; done; }
gpu_others() { local p q mine anc=" " a=$XE2_TOP
  while [[ -n $a && $a -gt 1 ]]; do anc+="$a "; a=$(ps -o ppid= -p $a 2>/dev/null | tr -d ' '); done
  for p in $( { grep -l -s "^drm-pdev:[[:space:]]*$PDEV" /proc/[0-9]*/fdinfo/* | cut -d/ -f3
                pgrep -f 'llama-server|ComfyUI|comfyui|ollama|vllm|llama_main|test_llama_microbench|Runner.Worker|custom_ops'; } | sort -un); do
    [[ $anc == *" $p "* ]] && continue
    monitor_idle $p && continue
    q=$p; mine=0
    while [[ -n $q && $q -gt 1 ]]; do [[ $q == "$XE2_TOP" ]] && { mine=1; break; }; q=$(ps -o ppid= -p $q 2>/dev/null | tr -d ' '); done
    [[ $mine == 0 && -d /proc/$p ]] && printf '%s:%s;' $p "$(ps -o comm= -p $p 2>/dev/null)"
  done; }
# kill_tree <pid>: stop a campaign job and every process descended from it (TERM, then KILL after 3 s). Only
# descendants of <pid> are signalled; a foreign process is never touched.
kill_tree() { local all=$1 new=$1 k
  while [[ -n $new ]]; do new=$(ps -o pid= --ppid "$(echo $new | tr ' ' ,)" 2>/dev/null | tr '\n' ' '); all+=" $new"; done
  kill -TERM $all 2>/dev/null; for k in 1 2 3; do sleep 1; kill -0 $all 2>/dev/null || return 0; done; kill -KILL $all 2>/dev/null; }
# guarded <others file> <command...>: refuse to start while a foreign GPU process runs; poll once a second
# during the command and, as soon as one appears, stop the command and its descendants (their logs stay as the
# interrupted evidence) and record the foreign process in the file. Returns the command's status, or 76 when a
# foreign process was seen before or during it.
guarded() { local f=$1 o j rc; shift
  o=$(gpu_others); [[ -n $o ]] && { echo "before start: $o" > "$f"; echo "other GPU process, not started: $o" >&2; return 76; }
  : > "$f"; "$@" & j=$!
  while kill -0 $j 2>/dev/null; do
    o=$(gpu_others)
    if [[ -n $o ]]; then
      echo "$(date -u +%T) $o" >> "$f"; kill_tree $j; wait $j 2>/dev/null
      echo "other GPU process during the job, job stopped: $o" >&2; return 76
    fi
    sleep 1
  done
  wait $j; rc=$?
  o=$(gpu_others); [[ -n $o ]] && { echo "$(date -u +%T) at end: $o" >> "$f"; echo "other GPU process at the end of the job: $o" >&2; return 76; }
  return $rc; }
