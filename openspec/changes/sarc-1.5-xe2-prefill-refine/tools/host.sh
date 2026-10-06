# host.sh: host constants and guards of the Xe2 campaign (fedora-gpu-eval), sourced by every tool here.
# XE2_CARD selects the card (default 0):
#   0 = b70-0, guest PCI 0000:01:00.0 = Vulkan device 0 (deviceUUID 868023e2-0000-0000-0100-000000000000). Every
#       selecting measurement, session, gate and reported number.
#   1 = the second Arc Pro B70, guest PCI 0000:02:00.0 = Vulkan device 1 (deviceUUID ...-0200-...; index and UUID
#       read with `vulkaninfo --summary`, and the index is checked against the DRM client of a running job by
#       card_test.sh). Cheap-mode screens only (owner decision 2026-10-05).
TOOLS=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")"); C=$(dirname $TOOLS)
ET=$(cd $C/../../.. && pwd); XE2_ROOT=$(dirname $ET)          # ~/hmz-sarc-xe2/executorch, ~/hmz-sarc-xe2
A=${XE2_ARTIFACTS:-$XE2_ROOT/.artifacts}
export XE2_CARD=${XE2_CARD:-0}
case $XE2_CARD in
  0) LOCK=868023e2-0000-0000-0100-000000000000; PDEV=0000:01:00.0;;
  1) LOCK=868023e2-0000-0000-0200-000000000000; PDEV=0000:02:00.0;;
  *) echo "XE2_CARD must be 0 or 1" >&2; exit 2;;
esac
PDEVS='0000:01:00.0|0000:02:00.0'; RUN=$A/run
PARENT_COMMIT=6a7cc8cc6
PCI=/sys/bus/pci/devices/$PDEV
HW=$(echo $PCI/hwmon/hwmon*); FREQ=$PCI/tile0/gt0/freq0
# Calibration written by `e2e5.sh --calibrate` (the baseline session) and required by every later session:
IDLE_FILE=$A/idle_temp_mc     # package temperature of the cool, idle card (millidegrees C)
CLKMIN_FILE=$A/clkmin_mhz     # lowest accepted median GT clock of a timed run (MHz)
export ETVK_DEVICE_INDEX=$XE2_CARD SARC_MOUNT_ROOT=$XE2_ROOT XE2_TOP=${XE2_TOP:-$$}
export XE2_PYTHON=${XE2_PYTHON:-$A/venv/bin/python}   # executorch.devtools for trace_analysis.py (a venv in the artifact directory)
gtemp_mc() { cat $HW/temp2_input; }   # package temperature, millidegrees C
# cool_start: wait (at most 5 min) until the package is within 3 C of the calibrated idle temperature, or has
# stopped falling (no drop over 30 s: the idle temperature of this card drifts between 56 and 64 C with its fan
# hysteresis); before the calibration exists, until the temperature has not moved by more than 1 C over 60 s.
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
  rm -f $idx; echo "$rel $sha" >> "${XE2_EXPORT_MANIFEST:?}"
  while read -r s p; do
    [[ -e $wt/$p/.git ]] || { echo "export: submodule $rel/$p is not initialised" >&2; return 1; }
    mkdir -p "$dest/$p" && export_commit "$wt/$p" "$s" "$dest/$p" "$rel/$p" || return 1
  done < <(git -C "$wt" ls-tree -r "$sha" | awk '$2 == "commit" {print $3, $4}'); }
# gpu_others: GPU workloads that this campaign job did not start, as "pid:command;" entries; empty = the card
# is ours. A process counts when
#   - it holds a DRM file of either B70 (any /proc/<pid>/fdinfo entry with drm-pdev in PDEVS), whatever its
#     name (a Vulkan process opens every card when it enumerates them, so the cards are not told apart here); or
#   - its command line names a known GPU workload. This second rule is the fallback for processes of other
#     users, whose fdinfo an unprivileged user cannot read (the fleet's LLM services run as this user).
# Ours = the top-level tool (XE2_TOP), its descendants, and its ancestors (the shells that launched it); and the
# jobs this campaign runs on the other card: each card's queue registers itself in run/queue<N>.top and every
# gl.sh in run/card<N>.job (pid and start time, so a reused pid does not count), and the descendants of a
# registered, living queue or gl.sh are ours on either card. (The queue registration is what covers a gl.sh
# that has started but not yet registered: the first split screen stopped on exactly that, 2026-10-05 23:00.)
# Not counted: an idle monitor (monitor_idle). The owner's nvtop was open on this host before the campaign
# started; it holds a DRM file of both cards to read their counters and submits nothing. It is exempt only
# while every DRM client it owns shows zero engine cycles and zero GPU memory; the moment either is non-zero it
# is a foreign GPU process like any other. Exempt monitors are listed by gpu_monitors and recorded per session.
monitor_idle() { [[ $(ps -o comm= -p $1 2>/dev/null) == nvtop ]] || return 1
  ! awk '/^drm-(cycles|total|resident|shared|active)-[a-z0-9]+:/ && $2 + 0 > 0 {f = 1} END {exit !f}' /proc/$1/fdinfo/* 2>/dev/null; }
gpu_monitors() { local p; for p in $(grep -l -s -E "^drm-pdev:[[:space:]]*($PDEVS)" /proc/[0-9]*/fdinfo/* | cut -d/ -f3 | sort -un); do
  monitor_idle $p && printf '%s:%s;' $p "$(ps -o comm= -p $p 2>/dev/null)"; done; }
# inline_shell <pid>: a shell running an inline command (sh -c "..."). Its command line is text, not a program
# name: the roofline run of 2026-10-04 21:46 was stopped because an operator shell's command text contained a
# watched name. Such a process is still caught by the DRM rule if it opens the card.
inline_shell() { local -a a; mapfile -d '' -t a < /proc/$1/cmdline 2>/dev/null || return 1
  [[ ${a[0]##*/} =~ ^(bash|sh|dash|zsh|fish)$ && ( ${a[1]:-} == -c || ${a[2]:-} == -c ) ]]; }
campaign_tops() { local f p s; for f in $RUN/card*.job $RUN/queue*.top; do [[ -e $f ]] || continue; read -r p s < $f || continue
  [[ -n $p && $(cut -d' ' -f22 /proc/$p/stat 2>/dev/null) == "$s" ]] && printf '%s ' $p; done; }
gpu_others() { local p q mine anc=" " a=$XE2_TOP tops; tops=" $XE2_TOP $(campaign_tops)"
  while [[ -n $a && $a -gt 1 ]]; do anc+="$a "; a=$(ps -o ppid= -p $a 2>/dev/null | tr -d ' '); done
  for p in $( { grep -l -s -E "^drm-pdev:[[:space:]]*($PDEVS)" /proc/[0-9]*/fdinfo/* | cut -d/ -f3
                for q in $(pgrep -f 'llama-server|ComfyUI|comfyui|ollama|vllm|llama_main|test_llama_microbench|Runner.Worker|custom_ops'); do inline_shell $q || echo $q; done; } | sort -un); do
    [[ $anc == *" $p "* ]] && continue
    monitor_idle $p && continue
    q=$p; mine=0
    while [[ -n $q && $q -gt 1 ]]; do [[ $tops == *" $q "* ]] && { mine=1; break; }; q=$(ps -o ppid= -p $q 2>/dev/null | tr -d ' '); done
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
# pair_lock shared|excl: the two cards share the VM's CPUs, memory bandwidth and power. A cheap-mode screen holds
# run/pair.lock shared (both cards may screen at once); everything else that measures or builds holds it
# exclusively (a full measurement, a session, a gate and a build each run with the other card idle and no
# build beside them). A waiting exclusive holder announces itself (run/excl-wanted.<pid>) and new shared holders
# wait for it, so two alternating screens cannot starve it. Held on fd 7 by the calling shell until it exits;
# a tool started under a holder (XE2_PAIR_HELD) does not take it again.
pair_wanted() { local f; for f in $RUN/excl-wanted.*; do [[ -e $f ]] || continue; [[ -d /proc/${f##*.} ]] && return 0; rm -f $f; done; return 1; }
pair_lock() { [[ -n ${XE2_PAIR_HELD:-} ]] && return 0; mkdir -p $RUN; exec 7>>$RUN/pair.lock
  if [[ $1 == shared ]]; then while pair_wanted; do sleep 2; done; flock -s 7
  else : > $RUN/excl-wanted.$BASHPID; flock -x 7; rm -f $RUN/excl-wanted.$BASHPID; fi
  XE2_PAIR_OWN=1; export XE2_PAIR_HELD=$1; }
# Coordinator hold (owner decision 2026-10-06): while the file HOLD exists in the artifact directory, nothing of
# this campaign starts on either card: no GPU process, no session, no gate, no build. Whatever is running is
# finished normally. The coordinator creates and removes HOLD; nobody else does.
#   coordinator_hold <what would start next>: returns at once without HOLD. Otherwise appends one line
#     `HELD <UTC time> card<N> <what would start next>` to the file HELD beside it, but only at a moment when
#     no measurement and no build of this campaign is running on either card (it must get run/pair.lock
#     exclusively for that moment, or be running under a tool that holds it), so HELD never appears while the
#     other card still works; then polls once a minute, and when HOLD is gone removes HELD and returns.
#     Every hold that was obeyed is recorded in <artifacts>/hold.log (the HELD line and when it was released).
#   gpu_begin shared|excl <what>: pair_lock, then the hold check WITH the lock held (so a job cannot slip in
#     between the check and its start); under a hold the lock is given back first. Every tool that measures or
#     builds starts with it: gl.sh before each GPU process (the smallest unit: one configuration), the session,
#     gate, trace, parent-control and roofline tools, build-sweep.sh, and the queue before each job.
HOLD=$A/HOLD; HELD=$A/HELD
coordinator_hold() { local said=0 line
  while [[ -e $HOLD ]]; do
    if [[ $said == 0 ]]; then line="HELD $(date -u +%FT%TZ) card$XE2_CARD $*"; mkdir -p $RUN
      if [[ -n ${XE2_PAIR_HELD:-} ]]; then echo "$line" >> $HELD; said=1
      elif ( exec 6>>$RUN/pair.lock; flock -n -x 6 && echo "$line" >> $HELD ); then said=1; fi
    fi
    if [[ $said == 1 ]]; then sleep ${XE2_HOLD_POLL:-60}; else sleep 5; fi
  done
  [[ $said == 1 ]] && { rm -f $HELD; echo "$line; released $(date -u +%FT%TZ)" >> $A/hold.log; }; return 0; }
gpu_begin() { local mode=$1; shift
  while :; do pair_lock $mode; [[ -e $HOLD ]] || return 0
    [[ -n ${XE2_PAIR_OWN:-} ]] && { exec 7>&-; unset XE2_PAIR_HELD XE2_PAIR_OWN; }
    coordinator_hold "$@"; done; }
