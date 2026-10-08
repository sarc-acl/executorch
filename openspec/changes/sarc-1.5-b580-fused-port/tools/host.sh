# host.sh: host constants and guards of the B580 fused-attention port (the owner's desktop `fedora`), sourced by
# every tool here. Copied from sarc-1.5-b580-prefill-refine/tools/host.sh; what differs: the parent commit (the
# tuned first-campaign head), the pristine commit of the closing session, and hold_wait (owner decision D6).
# The Arc B580 is PCI 0000:03:00.0 = Vulkan device 0 (deviceUUID 86800be2-0000-0000-0300-000000000000); Vulkan
# device 1 is the Ryzen iGPU and is never measured. Adapted from sarc-1.5-xe2-prefill-refine/tools/host.sh.
# What differs from the B70 host: the card drives the desktop, so it always has foreign DRM clients
# (gnome-shell, browser, terminal). A foreign DRM client is therefore not a reason to stop: the clients are
# recorded and their engine time is measured per run (tools/sampler.py, drm-cycles of the xe fdinfo), and a run
# disturbed beyond the calibrated level is rejected by e2e5.sh. Only a foreign GPU *workload* (a process whose
# program name is a known compute job) stops a job. And builds never overlap a GPU job: desktop-build lock.
TOOLS=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")"); C=$(dirname $TOOLS)
ET=$(cd $C/../../.. && pwd); B580_ROOT=$(dirname $ET)          # /mnt/linux-share/hmz-campaigns/b580-fused{/executorch,}
A=${B580_ARTIFACTS:-$B580_ROOT/.artifacts}
LOCK=86800be2-0000-0000-0300-000000000000
BUILD_LOCK=$HOME/.cache/gpu-lab/lock-desktop-build
PARENT_COMMIT=51d9d757f     # parent of every comparison, timed with PARENT_ENV
PARENT_ENV="ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-refine3"
PRISTINE_COMMIT=6a7cc8cc6   # the first campaign's pristine parent, for the closing session only
hold_wait() { "$TOOLS/hold.sh" wait "${1:-next unit}"; }   # blocks while <artifacts>/HOLD exists
PDEV=0000:03:00.0; PCI=/sys/bus/pci/devices/$PDEV
HW=$(echo $PCI/hwmon/hwmon*); FREQ=$PCI/tile0/gt0/freq0
# Calibration written by `e2e5.sh --calibrate` (the baseline / A-A session) and required by every later session:
IDLE_FILE=$A/idle_temp_mc     # package temperature of the cool, idle card (millidegrees C)
CLKMIN_FILE=$A/clkmin_mhz     # lowest accepted median GT clock of a timed run (MHz)
BUSYMAX_FILE=$A/busymax_pct   # highest accepted foreign engine time inside the timed prefill window (percent)
export ETVK_DEVICE_INDEX=0 SARC_MOUNT_ROOT=$B580_ROOT B580_TOP=${B580_TOP:-$$}
export B580_PYTHON=${B580_PYTHON:-$A/venv/bin/python}   # executorch.devtools for trace_analysis.py (a venv in the artifact directory)
export TMPDIR=${B580_TMPDIR:-$A/tmp}; mkdir -p $TMPDIR 2>/dev/null   # nothing large under /tmp or /home on this machine
gtemp_mc() { cat $HW/temp2_input; }   # package temperature, millidegrees C
# desktop_idle / idle_wait: the card drives the owner's desktop, and a desktop in use costs 4 to 10 % of the
# engine time of a run (s1-aa of 2026-10-08, superseded). A timed unit (a timed run, a trace run, a kernel screen
# run) therefore starts only while the graphical session of seat0 reports IdleHint=yes (no input for GNOME's idle
# delay, 900 s as found). Nothing is changed on the desktop; the wait is logged in logs/idle_wait.log. A run the
# owner's return disturbs is still caught by the foreign-engine-time limit (BUSYMAX).
desktop_idle() { [[ $(loginctl show-session "$(loginctl list-sessions --no-legend | awk '$4 == "seat0" && $6 == "user" {print $1; exit}')" -p IdleHint --value 2>/dev/null) == yes ]]; }
idle_wait() { local n=0
  until desktop_idle; do (( n++ % 45 == 0 )) && echo "$(date -u +%FT%TZ) desktop in use, waiting: ${1:-timed unit}" >> $A/logs/idle_wait.log; sleep 20; done
  (( n > 0 )) && echo "$(date -u +%FT%TZ) desktop idle after $((n * 20)) s: ${1:-timed unit}" >> $A/logs/idle_wait.log; return 0; }
# gpu_shared / build_exclusive: the desktop-build lock shared with the Jetson Orin campaign, which cross-builds
# on this machine. Every GPU job of this campaign holds it shared (fd 8, inherited by its children), every
# build holds it exclusive, so a build never runs during a measurement.
gpu_shared() { exec 8>>"$BUILD_LOCK"; flock -s -w ${1:-14400} 8 || { echo "desktop-build lock busy (a build is running)" >&2; return 75; }; }
build_exclusive() { exec 8>>"$BUILD_LOCK"; flock -x -w ${1:-14400} 8 || { echo "desktop-build lock busy (a measurement is running)" >&2; return 75; }; }
# cool_start: wait (at most 5 min) until the package is within 3 C of the calibrated idle temperature, or has
# stopped falling (no drop over 30 s); before the calibration exists, until the temperature has not moved by
# more than 1 C over 60 s.
cool_start() { local t0=$SECONDS a b
  if [[ -s $IDLE_FILE ]]; then
    while (( $(gtemp_mc) > $(<$IDLE_FILE) + 3000 && SECONDS - t0 < 300 )); do a=$(gtemp_mc); sleep 30; (( $(gtemp_mc) >= a )) && break; done
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
  rm -f $idx; echo "$rel $sha" >> "${B580_EXPORT_MANIFEST:?}"
  while read -r s p; do
    [[ -e $wt/$p/.git ]] || { echo "export: submodule $rel/$p is not initialised" >&2; return 1; }
    mkdir -p "$dest/$p" && export_commit "$wt/$p" "$s" "$dest/$p" "$rel/$p" || return 1
  done < <(git -C "$wt" ls-tree -r "$sha" | awk '$2 == "commit" {print $3, $4}'); }
# drm_clients: the DRM clients of the B580 that this job did not start, as "pid:command;" entries (one per
# process). Recorded per session and per run; on this desktop the list is never empty.
drm_clients() { local p q mine
  for p in $(grep -l -s "^drm-pdev:[[:space:]]*$PDEV" /proc/[0-9]*/fdinfo/* | cut -d/ -f3 | sort -un); do
    q=$p; mine=0
    while [[ -n $q && $q -gt 1 ]]; do [[ $q == "$B580_TOP" ]] && { mine=1; break; }; q=$(ps -o ppid= -p $q 2>/dev/null | tr -d ' '); done
    [[ $mine == 0 && -d /proc/$p ]] && printf '%s:%s;' $p "$(ps -o comm= -p $p 2>/dev/null | tr ' ,' '__')"
  done; 2>/dev/null; }
# gpu_others: GPU *workloads* that this campaign job did not start, as "pid:command;" entries; empty = no
# foreign compute job. A process counts when the name of the program it runs (argv[0], not the text of its
# command line: shells, ssh and build containers of the other campaigns carry these names as arguments) is a
# known GPU workload. Ours = the top-level tool (B580_TOP), its descendants and its ancestors. Desktop clients
# of the card are not listed here; their engine time is measured per run instead (drm_clients, sampler.py).
gpu_others() { local p q mine a0 anc=" " a=$B580_TOP
  while [[ -n $a && $a -gt 1 ]]; do anc+="$a "; a=$(ps -o ppid= -p $a 2>/dev/null | tr -d ' '); done
  for p in $(pgrep -f 'llama-server|ComfyUI|comfyui|ollama|vllm|llama_main|test_llama_microbench|logits_probe|roofline|custom_ops'); do
    [[ $anc == *" $p "* ]] && continue
    a0=$(tr '\0' '\n' < /proc/$p/cmdline 2>/dev/null | head -1); a0=${a0##*/}
    [[ $a0 =~ ^(llama-server|ollama|vllm|llama_main|test_llama_microbench|logits_probe|roofline)$ ]] \
      || { [[ $a0 =~ ^python ]] && tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -qE '^[^ ]+ +[^ ]*(ComfyUI|comfyui|vllm)'; } || continue
    q=$p; mine=0
    while [[ -n $q && $q -gt 1 ]]; do [[ $q == "$B580_TOP" ]] && { mine=1; break; }; q=$(ps -o ppid= -p $q 2>/dev/null | tr -d ' '); done
    [[ $mine == 0 && -d /proc/$p ]] && printf '%s:%s;' $p "$(ps -o comm= -p $p 2>/dev/null)"
  done 2>/dev/null; }   # a process that exits while it is examined must not write into the job's log
# kill_tree <pid>: stop a campaign job and every process descended from it (TERM, then KILL after 3 s). Only
# descendants of <pid> are signalled; a foreign process is never touched.
kill_tree() { local all=$1 new=$1 k
  while [[ -n $new ]]; do new=$(ps -o pid= --ppid "$(echo $new | tr ' ' ,)" 2>/dev/null | tr '\n' ' '); all+=" $new"; done
  kill -TERM $all 2>/dev/null; for k in 1 2 3; do sleep 1; kill -0 $all 2>/dev/null || return 0; done; kill -KILL $all 2>/dev/null; }
# guarded <others file> <command...>: refuse to start while a foreign GPU workload runs; poll once a second
# during the command and, as soon as one appears, stop the command and its descendants (their logs stay as the
# interrupted evidence) and record the foreign process in the file. Returns the command's status, or 76 when a
# foreign workload was seen before or during it.
guarded() { local f=$1 o j rc; shift
  o=$(gpu_others); [[ -n $o ]] && { echo "before start: $o" > "$f"; echo "other GPU workload, not started: $o" >&2; return 76; }
  : > "$f"; "$@" & j=$!
  while kill -0 $j 2>/dev/null; do
    o=$(gpu_others)
    if [[ -n $o ]]; then
      echo "$(date -u +%T) $o" >> "$f"; kill_tree $j; wait $j 2>/dev/null
      echo "other GPU workload during the job, job stopped: $o" >&2; return 76
    fi
    sleep 1
  done
  wait $j; rc=$?
  o=$(gpu_others); [[ -n $o ]] && { echo "$(date -u +%T) at end: $o" >> "$f"; echo "other GPU workload at the end of the job: $o" >&2; return 76; }
  return $rc; }
