# host.sh adapter for the Radeon 780M, for kit/session.sh. Written from the tuning campaign's own session
# script (sarc-1.5-780m-prefill-refine/tools/e2e5.sh and gl.sh): same lock, same sensors, same clock floor
# (2700 MHz median inside the prefill window), same list of GPU workloads that must not be running.
# The campaign samples every 0.1 s; a llama.cpp or tuned 1B prefill lasts 0.5 s here, so this adapter samples
# every 20 ms to keep at least five samples in the window. This device has no per-client engine accounting,
# so there is no foreign-busy ceiling: the session must run while the campaign's queue is held.
LOCK=00000000-c400-0000-0000-000000000000
HW=""; for h in /sys/class/hwmon/hwmon*; do [[ $(cat $h/name 2>/dev/null) == amdgpu ]] && HW=$h; done
GBUSY=$(ls /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | head -1)
DEV_CLKMIN=2700; DEV_BUSYMAX=""
export ETVK_DEVICE_INDEX=0
gtemp() { echo $(( $(<$HW/temp1_input) / 1000 )); }
gpu_shared() { :; }
drm_clients() { :; }
dev_sampler() { while :; do
    printf '%s %s %s %s %s\n' "${EPOCHREALTIME/./}" "$(<$HW/freq1_input)" "$(<$GBUSY)" "$(<$HW/power1_average)" "$(<$HW/temp1_input)"
    sleep 0.02; done > "$1" 2>/dev/null; }
# GPU workloads this session did not start, as "pid:command;" entries. Matched on the program name, as the
# campaign's gl.sh does: a held queue process that only carries such a name in its arguments is not a workload.
gpu_others() { local p q mine
  for p in $(pgrep -x 'llama-server|ollama|llama_main|test_llama_micr|llama-completio|llama-bench|vllm'; pgrep -f 'ComfyUI|comfyui' ); do
    q=$p; mine=0
    while [[ -n $q && $q -gt 1 ]]; do [[ $q == "$TOP" || $q == "$$" ]] && { mine=1; break; }; q=$(ps -o ppid= -p $q 2>/dev/null | tr -d ' '); done
    [[ $mine == 0 && -d /proc/$p ]] && printf '%s:%s;' $p "$(ps -o comm= -p $p 2>/dev/null)"
  done 2>/dev/null; }
# guarded <others file> <command...>: refuse to start beside a foreign GPU workload; record one seen at the end.
guarded() { local f=$1 o rc; shift
  o=$(gpu_others); [[ -n $o ]] && { echo "before start: $o" > "$f"; return 76; }
  : > "$f"; "$@"; rc=$?
  o=$(gpu_others); [[ -n $o ]] && { echo "$(date -u +%T) at end: $o" >> "$f"; return 76; }
  return $rc; }
