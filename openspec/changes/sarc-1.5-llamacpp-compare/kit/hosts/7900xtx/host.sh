# host.sh adapter for the Radeon RX 7900 XTX (AMDVLK), for kit/session.sh. The session runs ON the GPU host. Constants,
# sensors and rules are those of the tuning campaign (sarc-1.5-7900xtx-prefill-refine: tools/env.sh, others.sh, sampler.py,
# e2e5.sh, thresholds.txt): the same gpu-lab lock, Vulkan device 0, the campaign's ICD for every arm (ExecuTorch and
# llama.cpp), the maximum of the card's three hwmon temperatures, the calibrated clock floor (clkmin of thresholds.txt,
# 2670 MHz), the thermal rule (gpu_metrics indep_throttle_status bits 32 to 47 masked by thermal_mask, 0xffef: bit 36 is
# recorded, not rejecting; owner decision of the campaign), the pre-run foreign-busy bound (busy_pre_max, 5 %) and the
# campaign's foreign-user test (holders of the card's DRM nodes and GPU runner programs; the idle `ollama serve` of the
# host is not a runner).
# Sampling: the campaign's sampler.py every 5 ms (thresholds.txt), written in row.py's 6-column format
# (epoch_us sclk_MHz status energy_uJ temp_mC reasons): `status` = 1 when any throttle bit is set (recorded), `energy_uJ`
# = the integral of power1_average, `reasons` = "thermal" when the campaign's masked temperature bits are set, so that
# row.py judges the campaign's thermal rule inside the prefill window. The campaign's 7 columns go to <clock file>.full.
# Waits before every run, as the campaign's e2e5.sh: host builds of anyone (compilers, linkers) are waited out, then the
# card's gpu_busy_percent (the card drives the host's display) must be at most busy_pre_max in 10 samples of 50 ms; both
# waits are written to <run>.builds. There is no per-client engine accounting: no foreign-busy ceiling in row.py.
# Environment (not committed, host-specific):
#   VK_ICD_FILENAMES  the campaign's ICD, AMDVLK (required)
#   SARC7900_TOOLS    the campaign's tools/ directory (thresholds.txt); without it: clkmin 2670, mask 0xffef, busy 5
#   SARC7900_LOCK     the campaign's gpu-lab lock name (default: 7900xtx-<short host name>); session.sh puts the lock
#                     file under $HOME/.cache/gpu-lab, so run it with HOME=<a directory that has .cache/gpu-lab>
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../common-simple.sh"
[[ -n ${VK_ICD_FILENAMES:-} && -f ${VK_ICD_FILENAMES%%:*} ]] || { echo "host.sh: set VK_ICD_FILENAMES to the campaign's ICD (AMDVLK)" >&2; exit 2; }
export VK_ICD_FILENAMES ETVK_DEVICE_INDEX=0
LOCK=${SARC7900_LOCK:-7900xtx-$(hostname -s)}
CARD=/sys/class/drm/card1/device
[[ $(cat $CARD/vendor 2>/dev/null) == 0x1002 && $(cat $CARD/device 2>/dev/null) == 0x744c ]] || { echo "host.sh: card1 is not the RX 7900 XTX (1002:744c)" >&2; exit 70; }
HW=""; for h in /sys/class/hwmon/hwmon*; do
  [[ $(cat $h/name 2>/dev/null) == amdgpu && $(readlink -f $h/device) == $(readlink -f $CARD) ]] && HW=$h; done
[[ -n $HW ]] || { echo "host.sh: no amdgpu hwmon of card1" >&2; exit 70; }
THR=${SARC7900_TOOLS:-/nonexistent}/thresholds.txt
DEV_CLKMIN=$(sed -n 's/^clkmin=//p' "$THR" 2>/dev/null); DEV_CLKMIN=${DEV_CLKMIN:-2670}; DEV_BUSYMAX=""
TMASK=$(sed -n 's/^thermal_mask=//p' "$THR" 2>/dev/null); TMASK=${TMASK:-0xffef}
BUSYPRE=$(sed -n 's/^busy_pre_max=//p' "$THR" 2>/dev/null); BUSYPRE=${BUSYPRE:-5}
SPER=$(sed -n 's/^sample_period_s=//p' "$THR" 2>/dev/null); SPER=${SPER:-0.005}
gtemp() { local a b c
  { a=$(<$HW/temp1_input); b=$(<$HW/temp2_input); c=$(<$HW/temp3_input); } 2>/dev/null \
    || { echo "GPU sensors gone: stop, do not retry" >&2; kill -TERM $TOP; exit 70; }
  echo $(( (a > b ? (a > c ? a : c) : (b > c ? b : c)) / 1000 )); }
gtemp_core() { local a b; a=$(<$HW/temp1_input); b=$(<$HW/temp2_input); echo $(( (a > b ? a : b) / 1000 )); }
dev_sampler() { exec python3 -c '
import signal, struct, sys, time
out, H, C, per, mask = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5], 16)
run = [True]
def stop(*_): run[0] = False
signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
def rd(p):
    with open(p, "rb") as f: return f.read()
e_uj = 0.0; last = None
with open(out, "w", buffering=1) as o, open(out + ".full", "w", buffering=1) as g:
    while run[0]:
        try:
            t = time.time_ns() // 1000
            sclk = int(rd(H + "/freq1_input")); busy = int(rd(C + "/gpu_busy_percent")); pw = int(rd(H + "/power1_average"))
            temp = max(int(rd(H + "/temp%d_input" % i)) for i in (1, 2, 3))
            m = rd(C + "/gpu_metrics")
            thr = struct.unpack_from("<Q", m, 112)[0] if len(m) >= 120 else -1
            gfx = struct.unpack_from("<H", m, 40)[0] if len(m) >= 42 else -1
            if last is not None: e_uj += pw * (t - last) / 1e6
            last = t
            hot = thr >= 0 and ((thr >> 32) & mask) != 0
            o.write("%d %d %d %d %d %s\n" % (t, sclk // 1000000, 1 if thr > 0 else 0, e_uj, temp, "thermal" if hot else "-"))
            g.write("%d %d %d %d %d %x %d\n" % (t, sclk, busy, pw, temp, thr, gfx))
        except (OSError, ValueError):
            pass
        time.sleep(per)
' "$1" "$HW" "$CARD" "$SPER" "$TMASK"; }
# GPU users this session did not start, as "pid:command;": the campaign's others.sh test (holders of the card's DRM
# nodes that fuser can see, GPU runner programs by their own name, `ollama runner`; the idle `ollama serve` is not one)
# plus the kit's list, minus this session and its children.
gpu_others() { local p q mine seen=" "
  for p in $(fuser /dev/dri/renderD128 /dev/dri/card1 2>/dev/null) \
           $(pgrep -x 'llama-server|llama_main|test_llama_micr|llama-completio|llama-bench|llama-cli|vllm|logits_dump|vulkaninfo|igpu-roofline'; pgrep -f 'ollama runne[r]|ComfyUI|comfyui'); do
    [[ $seen == *" $p "* ]] && continue; seen+="$p "
    q=$p; mine=0
    while [[ -n $q && $q -gt 1 ]]; do [[ $q == "$TOP" || $q == "$$" ]] && { mine=1; break; }; q=$(ps -o ppid= -p $q 2>/dev/null | tr -d ' '); done
    [[ $mine == 0 && -d /proc/$p ]] && printf '%s:%s;' $p "$(ps -o comm= -p $p 2>/dev/null)"
  done 2>/dev/null; }
builds() { pgrep -l -x 'cc1|cc1plus|as|ld|ld.bfd|ld.gold|ld.lld|collect2|ninja|make|gmake|cmake|ccache|clang|clang\+\+|gcc|g\+\+|c\+\+|glslc|rustc' | head -5 | tr ' \n' ':;'; }
busy_pre() { local i x bp=0; for i in 1 2 3 4 5 6 7 8 9 10; do x=$(<$CARD/gpu_busy_percent); (( x > bp )) && bp=$x; sleep 0.05; done; echo $bp; }
# guarded <others file> <command...>: waits out host builds and a busy card (at most 1800 s each; the card's display
# is a foreign user the campaign waited for, never forced), then the kit's guard. Waits and builds in <run>.builds.
guarded() { local f=$1 o rc t0=$SECONDS bw=0 fw=0 bp; shift
  while [[ -n $(builds) ]] && (( SECONDS - t0 < 1800 )); do sleep 5; done; bw=$((SECONDS - t0)); t0=$SECONDS
  bp=$(busy_pre); while (( bp > BUSYPRE && SECONDS - t0 < 1800 )); do sleep 2; bp=$(busy_pre); done; fw=$((SECONDS - t0))
  echo "before: $(builds) buildwait_s=$bw busy_pre=$bp busywait_s=$fw" > "${f%.others}.builds"
  [[ -n $(builds) ]] && { echo "before start: host build $(builds)" > "$f"; return 76; }
  (( bp > BUSYPRE )) && { echo "before start: card busy ${bp}%" > "$f"; return 76; }
  o=$(gpu_others); [[ -n $o ]] && { echo "before start: $o" > "$f"; return 76; }
  : > "$f"; "$@"; rc=$?
  echo "after: $(builds)" >> "${f%.others}.builds"
  o=$(gpu_others); [[ -n $o ]] && { echo "$(date -u +%T) at end: $o" >> "$f"; return 76; }
  return $rc; }
