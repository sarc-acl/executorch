# host.sh adapter for the Radeon RX 7600, for kit/session.sh. Constants and sensors are those of the tuning
# campaign (sarc-1.5-rx7600-prefill-refine, tools/env.sh, others.sh, sampler.py, thresholds.txt): the same gpu-lab
# lock, Vulkan device 0, the user-space RADV ICD for every arm (ExecuTorch and llama.cpp), the maximum of the
# card's three hwmon temperatures, the calibrated clock floor (clkmin of thresholds.txt, 2420 MHz), and the
# campaign's foreign-user test (holders of the card's DRM nodes and GPU runner programs).
# The campaign's sampler writes 7 columns, which row.py does not read; this adapter samples the same sysfs files
# every 10 ms and writes row.py's 5-column AMD format (epoch_us sclk_Hz busy_pct power_uW temp_mC) to the clock
# file, plus the campaign's 7 columns (with indep_throttle_status from gpu_metrics) to <clock file>.full, so the
# campaign's thermal rule can be checked afterwards. There is no per-client engine accounting: no foreign-busy
# ceiling. Host builds of anyone are not judged by row.py; they are recorded per run in <run>.builds.
# Environment (not committed, host-specific):
#   VK_ICD_FILENAMES  the campaign's RADV ICD (required)
#   RX7600_TOOLS      the campaign's tools/ directory (for thresholds.txt); without it the floor is 2420 MHz
#   RX7600_LOCK       the campaign's gpu-lab lock name (default: rx7600-<first field of the short host name>)
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../common-simple.sh"
[[ -n ${VK_ICD_FILENAMES:-} && -f ${VK_ICD_FILENAMES%%:*} ]] || { echo "host.sh: set VK_ICD_FILENAMES to the campaign's RADV ICD" >&2; exit 2; }
export VK_ICD_FILENAMES ETVK_DEVICE_INDEX=0
LOCK=${RX7600_LOCK:-rx7600-$(hostname -s | cut -d- -f1)}
CARD=/sys/class/drm/card1/device
HW=$(echo $CARD/hwmon/hwmon*)
[[ $(cat $HW/name 2>/dev/null) == amdgpu ]] || { echo "host.sh: no amdgpu hwmon under card1" >&2; exit 70; }
DEV_CLKMIN=$(sed -n 's/^clkmin=//p' "${RX7600_TOOLS:-/nonexistent}/thresholds.txt" 2>/dev/null); DEV_CLKMIN=${DEV_CLKMIN:-2420}; DEV_BUSYMAX=""
gtemp() { local a b c
  { a=$(<$HW/temp1_input); b=$(<$HW/temp2_input); c=$(<$HW/temp3_input); } 2>/dev/null \
    || { echo "GPU sensors gone: stop, do not retry" >&2; kill -TERM $TOP; exit 70; }
  echo $(( (a > b ? (a > c ? a : c) : (b > c ? b : c)) / 1000 )); }
dev_sampler() { exec python3 -c '
import signal, struct, sys, time
out, H, C = sys.argv[1], sys.argv[2], sys.argv[3]
run = [True]
def stop(*_): run[0] = False
signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
def rd(p):
    with open(p, "rb") as f: return f.read()
with open(out, "w", buffering=1) as o, open(out + ".full", "w", buffering=1) as g:
    while run[0]:
        try:
            t = time.time_ns() // 1000
            sclk = int(rd(H + "/freq1_input")); busy = int(rd(C + "/gpu_busy_percent")); pw = int(rd(H + "/power1_average"))
            temp = max(int(rd(H + "/temp%d_input" % i)) for i in (1, 2, 3))
            m = rd(C + "/gpu_metrics")
            thr = struct.unpack_from("<Q", m, 112)[0] if len(m) >= 120 else -1
            gfx = struct.unpack_from("<H", m, 40)[0] if len(m) >= 42 else -1
            o.write("%d %d %d %d %d\n" % (t, sclk, busy, pw, temp))
            g.write("%d %d %d %d %d %x %d\n" % (t, sclk, busy, pw, temp, thr, gfx))
        except (OSError, ValueError):
            pass
        time.sleep(0.01)
' "$1" "$HW" "$CARD"; }
# GPU users this session did not start, as "pid:command;": the campaign's others.sh test (holders of the card's DRM
# nodes, runner programs by their own name) plus the kit's list, minus this session and its children.
gpu_others() { local p q mine seen=" "
  for p in $(fuser /dev/dri/renderD128 /dev/dri/card1 2>/dev/null) \
           $(pgrep -x 'llama-server|ollama|llama_main|test_llama_micr|llama-completio|llama-bench|llama-cli|vllm|logits_dump|vulkaninfo'; pgrep -f 'ComfyUI|comfyui'); do
    [[ $seen == *" $p "* ]] && continue; seen+="$p "
    q=$p; mine=0
    while [[ -n $q && $q -gt 1 ]]; do [[ $q == "$TOP" || $q == "$$" ]] && { mine=1; break; }; q=$(ps -o ppid= -p $q 2>/dev/null | tr -d ' '); done
    [[ $mine == 0 && -d /proc/$p ]] && printf '%s:%s;' $p "$(ps -o comm= -p $p 2>/dev/null)"
  done 2>/dev/null; }
builds() { pgrep -l -x 'cc1|cc1plus|as|ld|ld.bfd|ld.gold|ld.lld|collect2|ninja|make|gmake|cmake|ccache|clang|clang\+\+|gcc|g\+\+|c\+\+|glslc|rustc' | head -5 | tr ' \n' ':;'; }
# guarded <others file> <command...>: the kit's guard, plus host builds seen before and after the run in <run>.builds.
guarded() { local f=$1 o rc; shift
  echo "before: $(builds)" > "${f%.others}.builds"
  o=$(gpu_others); [[ -n $o ]] && { echo "before start: $o" > "$f"; return 76; }
  : > "$f"; "$@"; rc=$?
  echo "after: $(builds)" >> "${f%.others}.builds"
  o=$(gpu_others); [[ -n $o ]] && { echo "$(date -u +%T) at end: $o" >> "$f"; return 76; }
  return $rc; }
