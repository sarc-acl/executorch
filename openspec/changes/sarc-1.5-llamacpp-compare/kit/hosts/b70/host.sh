# host.sh adapter for the Arc Pro B70 (card b70-0), for kit/session.sh. Constants and sensors are those of the
# tuning campaign's tools/host.sh and sampler.py (lock, PCI device 0000:01:00.0 = Vulkan device 0, package
# temperature, the calibrated clock floor in <artifacts>/clkmin_mhz). The campaign's own host.sh is not sourced:
# its tools wait while the coordinator hold exists, and this session is what the hold is for. The second card
# stays idle during the session (the campaign is held on both).
XE2=${XE2_ROOT:-$HOME/hmz-sarc-xe2}
LOCK=868023e2-0000-0000-0100-000000000000
PCI=/sys/bus/pci/devices/0000:01:00.0; HW=$(echo $PCI/hwmon/hwmon*)
DEV_CLKMIN=$(cat $XE2/.artifacts/clkmin_mhz); DEV_BUSYMAX=""
export ETVK_DEVICE_INDEX=0
gtemp() { echo $(( $(<$HW/temp2_input) / 1000 )); }
gpu_shared() { :; }
drm_clients() { :; }
dev_sampler() { exec python3 $XE2/executorch/openspec/changes/sarc-1.5-xe2-prefill-refine/tools/sampler.py "$1" 0.005; }
# GPU workloads this session did not start, as "pid:command;" entries, matched on the program name.
gpu_others() { local p q mine
  for p in $(pgrep -x 'llama-server|ollama|llama_main|test_llama_micr|llama-completio|llama-bench|vllm'; pgrep -f 'ComfyUI|comfyui'); do
    q=$p; mine=0
    while [[ -n $q && $q -gt 1 ]]; do [[ $q == "$TOP" || $q == "$$" ]] && { mine=1; break; }; q=$(ps -o ppid= -p $q 2>/dev/null | tr -d ' '); done
    [[ $mine == 0 && -d /proc/$p ]] && printf '%s:%s;' $p "$(ps -o comm= -p $p 2>/dev/null)"
  done 2>/dev/null; }
guarded() { local f=$1 o rc; shift
  o=$(gpu_others); [[ -n $o ]] && { echo "before start: $o" > "$f"; return 76; }
  : > "$f"; "$@"; rc=$?
  o=$(gpu_others); [[ -n $o ]] && { echo "$(date -u +%T) at end: $o" >> "$f"; return 76; }
  return $rc; }
