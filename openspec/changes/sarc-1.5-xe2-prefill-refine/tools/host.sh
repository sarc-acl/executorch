# host.sh: host constants and guards of the Xe2 campaign (fedora-gpu-eval, card b70-0), sourced by every tool here.
# b70-0 is guest PCI 0000:01:00.0 = Vulkan device 0 (deviceUUID 868023e2-0000-0000-0100-000000000000).
XE2_ROOT=$HOME/hmz-sarc-xe2
A=${XE2_ARTIFACTS:-$XE2_ROOT/.artifacts}; ET=$XE2_ROOT/executorch
C=$ET/openspec/changes/sarc-1.5-xe2-prefill-refine; TOOLS=$C/tools
LOCK=868023e2-0000-0000-0100-000000000000
PARENT_COMMIT=6a7cc8cc6
PCI=/sys/bus/pci/devices/0000:01:00.0
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
# gpu_others: known GPU workloads that this campaign job did not start, as "pid:command;" entries; empty = the
# card is ours. Ours = the top-level tool (XE2_TOP), its descendants, and its ancestors (the shells that launched
# it, whose command lines may name the binaries).
gpu_others() { local p q mine anc=" " a=$XE2_TOP
  while [[ -n $a && $a -gt 1 ]]; do anc+="$a "; a=$(ps -o ppid= -p $a 2>/dev/null | tr -d ' '); done
  for p in $(pgrep -f 'llama-server|ComfyUI|comfyui|ollama|vllm|llama_main|test_llama_microbench|Runner.Worker|custom_ops'); do
    [[ $anc == *" $p "* ]] && continue
    q=$p; mine=0
    while [[ -n $q && $q -gt 1 ]]; do [[ $q == "$XE2_TOP" ]] && { mine=1; break; }; q=$(ps -o ppid= -p $q 2>/dev/null | tr -d ' '); done
    [[ $mine == 0 && -d /proc/$p ]] && printf '%s:%s;' $p "$(ps -o comm= -p $p 2>/dev/null)"
  done; }
# others_watch <file> & : append every foreign GPU process seen, once a second, until killed.
others_watch() { local o; : > "$1"; while :; do o=$(gpu_others); [[ -n $o ]] && echo "$(date -u +%T) $o" >> "$1"; sleep 1; done; }
# guarded <others file> <command...>: refuse to start while a foreign GPU process runs, watch for one during the
# command. Returns the command's status, or 76 when a foreign process was seen (the file then lists it).
guarded() { local f=$1 o w rc; shift
  o=$(gpu_others); [[ -n $o ]] && { echo "before start: $o" > "$f"; echo "other GPU process, not started: $o" >&2; return 76; }
  others_watch "$f" 9>&- & w=$!
  "$@"; rc=$?
  kill $w 2>/dev/null; wait $w 2>/dev/null
  [[ -s $f ]] && { echo "other GPU process during the job: $(head -1 "$f")" >&2; return 76; }
  return $rc; }
