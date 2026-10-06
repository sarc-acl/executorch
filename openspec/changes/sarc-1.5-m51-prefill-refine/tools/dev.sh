#!/bin/bash
# dev.sh: sourced by every tool of this campaign. Board, paths and the device-state guard.
#   M51_HOST, M51_SERIAL, M51_TUNNEL name the board, M51_MD5 its driver, M51_*_KHZ its clock pins; all from the
#   campaign's local settings file, never from this tree.
#   ART = artifact directory; DEV_ROOT = work directory on the board.
CFG=${M51_CFG:-$HOME/.config/sarc-m51.env}
[[ -f $CFG ]] && source "$CFG"
: "${M51_HOST:?set M51_HOST in $CFG}" "${M51_SERIAL:?set M51_SERIAL in $CFG}" "${M51_MD5:?set M51_MD5 in $CFG}"
: "${M51_GPU_KHZ:?}" "${M51_MIF_KHZ:?}" "${M51_INT_KHZ:?}"
export ADB_SERVER_SOCKET=${M51_TUNNEL:-tcp:localhost:5038}
ART=${ART:-<artifacts>}; LOC=m51-LOCAL-ONLY
DEV_ROOT=${DEV_ROOT:-<device-dir>}
TOOLS=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
A() { adb -s "$M51_SERIAL" "$@"; }
alive() { A get-state 2>/dev/null | grep -q device; }
gtemp() { A shell 'cat /sys/class/thermal/thermal_zone4/temp' 2>/dev/null | tr -d '\r' | awk '{printf "%d", $1/1000}'; }
clocks() { A shell 'for f in 23400000.sgpu/min_freq 23400000.sgpu/max_freq 17000010.devfreq_mif/cur_freq 17000020.devfreq_int/cur_freq; do echo -n "$f=$(cat /sys/class/devfreq/$f) "; done' 2>/dev/null | tr -d '\r'; }
# device_state: prints "ok" or the first reason the board is not fit for a GPU job.
device_state() {
  [[ -e $ART/GPU_GONE || -e $ART/ABORTED ]] && { echo "marker $(ls $ART/GPU_GONE $ART/ABORTED 2>/dev/null)"; return; }
  alive || { echo device_gone; return; }
  local h c; h=$(A shell md5sum /vendor/lib64/hw/vulkan.samsung.so 2>/dev/null | cut -d' ' -f1)
  [[ $h == "$M51_MD5" ]] || { echo "driver_md5"; return; }
  A shell '[ -e /data/vendor/gpu/amdPalSettings.cfg ] && echo PRESENT' 2>/dev/null | grep -q PRESENT && { echo pal_cfg_on; return; }
  c=$(clocks)
  [[ $c == *"sgpu/min_freq=$M51_GPU_KHZ "* && $c == *"sgpu/max_freq=$M51_GPU_KHZ "* && $c == *"mif/cur_freq=$M51_MIF_KHZ "* && $c == *"int/cur_freq=$M51_INT_KHZ "* ]] \
    || { echo "clock_pin ($c)"; return; }
  echo ok
}
# gpu_others: other processes on the board that use the GPU (none expected: only our runner and microbench).
gpu_others() { A shell 'ps -A -o PID,NAME' 2>/dev/null | tr -d '\r' | awk 'NR>1 && $2 ~ /llama_main|test_llama_microbench|igpu|vkcube|benchmark/ {printf "%s:%s;", $1, $2}'; }
# other_timed_session: true while a timed session runs, another campaign's on this workstation or one of this
# campaign's on the board (R5: no build then).
other_timed_session() {
  pgrep -f 'tools/e2e_m51\.sh --stage' > /dev/null ||
  { [[ -n ${OTHER_TIMED_SESSION:-} ]] && pgrep -f "$OTHER_TIMED_SESSION" > /dev/null; } ||
    { [[ -n ${OTHER_GPU_LOCK:-} ]] && lslocks -n -o PATH 2>/dev/null | grep -q "$OTHER_GPU_LOCK"; }
}
